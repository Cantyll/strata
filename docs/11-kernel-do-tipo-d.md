# Kernel do tipo D na velocidade do tipo P

Plano e régua escritos em 2026-10-27, **antes de medir qualquer velocidade**. Pergunta: o kernel CUDA dos tipos D (`STRATA_Q4_K_D1/D2`, `STRATA_Q6_K_D1/D2`, ggml 47 a 50) consegue gerar tokens a ≥ 0,95 da velocidade dos tipos P (43 a 46) sem perder a qualidade que os torna interessantes?

## Ponto de partida

- **Qualidade passou.** No Llama-3.1-8B, os tipos D com os bits escolhidos pelo GPTQ passam a régua de qualidade nos dois níveis (nível 1 −0,066 / +0,002 bit/peso, nível 2 −0,038 / +0,021 contra a curva dos k-quants com imatrix, wikitext / C4; PR #15, `docs/10`). Este trabalho não mexe no formato nem no compressor.
- **Velocidade reprovou.** Seção 30 de [resultados](05-resultados.md#30-sub-células-por-tensor-no-llama-31-8b-tipos-d-na-gpu), A100: D gera 94,1 / 93,1 tok/s contra 145,4 / 142,2 do P (0,65 nos dois níveis); contra o k-quant com imatrix, 0,64 (Q5_K_M, 146,0) e 0,79 (Q6_K, 118,6). Na CPU, 0,73 a 0,81 do P (seção 29).
- **O que não se repete:** medir a qualidade dos tipos D com bits GPTQ (já feito, PR #15) e a ideia de deslocamento por código (a "borda" dá a mesma KL e cabe num registrador, seção 30).

## Diagnóstico

A hipótese era "o gargalo é a consulta à tabela". O assembly (SASS do `mul_mat_vec_q` compilado para sm_80, o A100) mostra que a consulta em si é barata e que o custo está em tudo o que vem antes dela.

**O MMVQ do A100 não está preso na memória.** O tipo P gera 145 tok/s lendo ~5,5 GiB por token: ~0,85 TB/s, menos da metade dos ~2 TB/s do A100-80GB. Neste regime, o tempo de cada matriz acompanha o número de instruções do laço interno. Analogia: a esteira (memória) entrega as caixas mais rápido do que o operário (as unidades de conta) consegue abri-las, então cada gesto a mais do operário aparece no relógio.

Instruções no laço interno do MMVQ (uma coluna, sem fusão), contadas com `cuobjdump -sass`:

| tipo | antes | depois | depois, tabela linha | tipo P de mesma largura |
|---|---:|---:|---:|---:|
| Q4_K_D1 (nível 1, 81% dos pesos) | 228 | **149** | **109** | 104 |
| Q4_K_D2 (nível 2, 81% dos pesos) | 227 | **176** | **126** | 106 |
| Q6_K_D1 (nível 1, 19% dos pesos) | 245 | 198 | 152 | 150 |
| Q6_K_D2 (nível 2, 19% dos pesos) | 191 | 140 | 78 | 105 |

(Os laços dos tipos derivados do Q6_K não cobrem sempre o mesmo número de pesos por volta: o compilador desenrola de jeitos diferentes. Para eles, só o tempo medido vale.)

O kernel antigo gastava 2,2 vezes as instruções do P por peso nos tipos Q4_K. A conta extra vinha de três lugares:
1. a classe do código do nível 0 (menor, maior ou meio) por dois `__vcmpeq4` e máscaras;
2. o índice da tabela montado byte a byte e um `__byte_perm` duplo para cada 4 pesos;
3. um segundo `dp4a` com `δ` e a soma do código do nível 0 refeita à parte.

## O que mudou no kernel

- **Desvio do centro.** O código efetivo do tipo D é `T = 2^(k+4)·c + δ`. Ele vira `T = 16·u + 8(1 − 2^k) + e`, onde `u` é o código que o tipo P já monta e `e = δ − 8(2r + 1 − 2^k)` é o quanto a sub-célula do D se afasta do centro da do P. Assim o kernel D é o kernel P mais uma consulta: a soma de `u` é a mesma do P, e só `e` é novo. O host monta a tabela de `e` uma vez por tensor. A conta é inteira e dá o mesmo número que antes.
- **Classe por somas.** `c ≥ 1` é o vai-um do bit k+4 em `u + (2^(k+4) − 2^k)`; `c = 15` é o vai-um de `u + 2^k`. Duas somas e duas máscaras para 4 pesos, no lugar de duas comparações e da montagem do índice.
- **Oito consultas num `prmt`.** Os índices de 8 pesos vão empacotados em meio byte cada, e um `prmt.b32` em PTX consulta os 8 de uma vez (`strata_tab_8`).
- **Fusão gate+up ligada.** A tabela do gate vai num segundo campo de `ggml_cuda_mm_fusion_args_device`; antes, a fusão ficava desligada para os tipos D, e o P a usava.
- **Tabela linha (a "variante de pontos").** Quando as três classes da tabela de um tensor são iguais, a classe não muda nada e o host escolhe um kernel sem classe (pseudo-tipos 57 a 60, só dentro do CUDA). Ele é exato para essa tabela. No k = 1, o desvio é afim no bit r e a consulta vira um `dp4a`. O gravador ganha `--sem-classe` em `tests/kernel/strata_gguf_niveis.py`: o Lloyd ajusta uma tabela sem classe, ou seja, o mesmo deslocamento para todo código do nível 0. A qualidade dela não é conhecida e é medida aqui.

Conferências antes de ir à GPU: os kernels CUDA compilados como C++ na CPU dão a mesma conta que a referência em todos os tipos, com e sem tabela linha (`tests/kernel/strata_cuda_host.sh`; uma mutação proposital na tabela é pega). O `mmvq.cu` compila com o nvcc 13 para sm_80. `tests/kernel/strata_cuda_check.c` agora roda cada tipo D também com uma tabela linha.

## Previsão (estimativa, não resultado)

Um modelo linear `tempo = fixo + custo·instruções`, ajustado aos tempos da seção 30, prevê:

| | D exato (borda) | D linha |
|---|---:|---:|
| nível 1, / P | ~0,83 | ~0,98 |
| nível 1, / Q5_K_M | ~0,82 | ~0,97 |
| nível 2, / P | ~0,77 | ~0,92 |
| nível 2, / Q6_K | ~0,92 | ~1,10 |

O modelo não conta a fusão, que deve somar alguns por cento aos dois. Leitura honesta da previsão: **o D exato provavelmente não chega a 0,95 do P.** A classe mais a consulta custam ~6 instruções por 4 pesos, e o P inteiro tem ~26. A rota para 0,95 é a tabela linha, se a qualidade dela aguentar.

**Risco da tabela linha, visto antes de medir.** Num teste com pesos gaussianos sintéticos (2000 × 256 pesos, imatrix aleatória), o Lloyd sem classe devolve no nível 1 exatamente os centros do P (−8 e +8), e o erro ponderado sobe 10% contra a tabela com classe (no nível 2, 9%). Ou seja: nesse teste, todo o ganho do D sobre o P vem das duas bordas (códigos 0 e 15 do Q4_K, 0 e 63 do Q6_K), onde os pesos só caem de um lado, e o deslocamento do meio não ganha nada. Se os pesos reais forem assim, o D linha vai dar a KL do P e reprovar na qualidade. Por isso o notebook mede também o P e mostra quanto do ganho do D o linha guarda.

## Régua (definida antes de medir)

**Onde:** Colab A100, Llama-3.1-8B (`unsloth/Meta-Llama-3.1-8B`), o mesmo modelo da seção 30 e do PR #15. Nada de conclusão a partir do 1B.

**Velocidade:** `llama-bench -p 0 -n 128 -r 5`, geração de 128 tokens. Três rodadas intercaladas (todos os modelos e binários numa rodada, depois a próxima), mediana das três. "Antes" é o llama.cpp com o patch do commit `f441834` (kernel antigo, mesmo patch consertado); "depois" é o deste PR. Os mesmos GGUFs rodam nos dois.

| veredito | condição |
|---|---|
| **passa** | D ≥ 0,95 × P **e** D ≥ 0,95 × k-quant equivalente (Q5_K_M com imatrix no nível 1, Q6_K com imatrix no nível 2), nos dois níveis |
| **zona cinza** | a menor das quatro razões fica entre 0,85 e 0,95, ou só um nível passa |
| **reprova** | D < 0,85 × P em algum nível |

O veredito sai separado para o **D exato** (tabela com classes, a que passou na qualidade no PR #15) e para o **D linha**.

**Qualidade:**
- **D exato:** a conta é a mesma, então a KL pelo MMVQ (`llama-perplexity -ub 8`, que força o produto escalar da geração) do "antes" e do "depois" tem que bater dentro do ±. `strata_cuda_check` tem que terminar em `tudo OK`.
- **D linha:** passa se a KL dele ficar ≤ 1,03 × a do D exato nos dois níveis, com os mesmos bits do nível 0 e a mesma imatrix, no caminho da desquantização (o mesmo da régua de qualidade). O notebook mostra também a fração do ganho do D sobre o P que o linha guarda, (KL_P − KL_linha) / (KL_P − KL_D). Se reprovar, o D linha só vale como um ponto mais rápido e pior, e a próxima pergunta vai para o compressor: o GPTQ escolhendo bits para a tabela linha recupera a diferença?

**Atribuição:** o "depois" roda também com `GGML_CUDA_DISABLE_FUSION=1`, para separar o ganho da fusão do ganho do kernel.

**Prompt (pp512):** fica fora da régua. Ele passa pela desquantização + cuBLAS, que este trabalho não muda.

## Como medir

[`tests/kernel/strata_8b_kernel_d_colab.ipynb`](../tests/kernel/strata_8b_kernel_d_colab.ipynb): compila os dois llama.cpp (antes e depois), roda `strata_cuda_check`, grava P, D e D linha (níveis 1 e 2) sobre o mesmo Q4_K_M com imatrix, mede a KL e a velocidade e imprime o veredito desta régua.

## Resultado

A preencher com os números do notebook.
