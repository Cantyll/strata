# Kernel do tipo D na velocidade do tipo P

Plano e régua escritos em 2026-10-27, **antes de medir qualquer velocidade**. Pergunta: o kernel CUDA dos tipos D (`STRATA_Q4_K_D1/D2`, `STRATA_Q6_K_D1/D2`, ggml 47 a 50) consegue gerar tokens a ≥ 0,95 da velocidade dos tipos P (43 a 46) sem perder a qualidade que os torna interessantes?

## Ponto de partida

- **Qualidade passou.** No Llama-3.1-8B, os tipos D com os bits escolhidos pelo GPTQ passam a régua de qualidade nos dois níveis (nível 1 −0,066 / +0,002 bit/peso, nível 2 −0,038 / +0,021 contra a curva dos k-quants com imatrix, wikitext / C4; PR #15, `docs/10`). Este trabalho não mexe no formato nem no compressor.
- **Correção (2026-10-29, thread do compressor):** o "D com GPTQ passa" do PR #15 usava um deslocamento por código do nível 0, não a tabela do kernel (3 classes vezes o caminho). O número não vale para o D que roda. No 1B, com a tabela do kernel e bits GPTQ, o D linha empata com o P com GPTQ (KL ×1000 9,17/3,17 contra 8,52/3,17 no wikitext, níveis 1/2). O 8B decide (`strata_8b_gptq_linha_colab.ipynb`, branch do PR #15).
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

## Resultado (Colab A100-SXM4-80GB, 2026-10-27)

Commit medido `256d05e` (o `9cfb789` mais o conserto do assert dos tipos LINHA). Tabelas completas em [`resultados/unsloth__Meta-Llama-3.1-8B_kernel_d_a100.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_kernel_d_a100.md). `strata_cuda_check`: tudo OK, inclusive com tabela linha. As três rodadas variaram menos de 0,5%.

| tok/s (tg128) | depois | depois sem fusão | antes |
|---|---:|---:|---:|
| Q5_K_M imatrix | 147,6 | | |
| Q6_K imatrix | 120,8 | | |
| P nível 1 / 2 | 142,4 / 147,3 | 138,7 / 137,5 | 136,1 / 142,6 |
| D exato nível 1 / 2 | **125,3 / 120,1** | 121,6 / 113,1 | 92,0 / 93,5 |
| D linha nível 1 / 2 | **136,1 / 143,5** | 135,9 / 132,6 | |

| candidato | N1 / P | N1 / Q5_K_M | N2 / P | N2 / Q6_K | veredito |
|---|---:|---:|---:|---:|---|
| D exato, antes | 0,68 | 0,62 | 0,66 | 0,77 | reprova |
| D exato, depois | 0,88 | 0,85 | 0,82 | 0,99 | **reprova** |
| D linha, depois | 0,96 | 0,92 | 0,97 | 1,19 | **zona cinza** |

| qualidade (KL ×1000, caminho da desquantização, 80 trechos) | nível 1 | nível 2 |
|---|---:|---:|
| P | 11,26 ± 0,24 | 3,82 ± 0,08 |
| D exato | 9,70 ± 0,21 | 3,53 ± 0,07 |
| D linha | 9,79 ± 0,23 | 3,57 ± 0,07 |
| D linha / D exato (régua ≤ 1,03) | **1,009** | **1,012** |
| fração do ganho do D sobre o P que o linha guarda | 0,94 | 0,85 |
| D antes → depois pelo MMVQ (`-ub 8`) | 9,80 → 9,82 (+0,08 ±) | 3,76 → 3,75 (−0,10 ±) |

Leitura:

- **O kernel novo é 36% (nível 1) e 28% (nível 2) mais rápido que o antigo** e dá a mesma KL. A previsão pelo número de instruções acertou o nível 1 (0,83 previsto, 0,88 medido) e o nível 2 (0,77, 0,82). A fusão gate+up soma 3% a 6%. Mesmo assim, o D exato reprova: 0,82 do P no nível 2, abaixo de 0,85. A conta da classe custa caro demais para o P, que é quase só leitura e `dp4a`.
- **O D linha passou na qualidade, e o risco do teste sintético não apareceu.** Nos pesos reais, o deslocamento do meio ganha alguma coisa: a tabela sem classe guarda 94% (nível 1) e 85% (nível 2) do ganho do D sobre o P, e a KL fica 1% acima da do D exato.
- **Na velocidade, o D linha fica na zona cinza por uma razão só:** 0,92 do Q5_K_M no nível 1. Contra o P ele passa (0,96 e 0,97), e no nível 2 é 19% mais rápido que o Q6_K. O próprio P só chega a 0,965 do Q5_K_M no nível 1, então passar ali exige o linha a ≥ 0,984 do P.
- **Uma anomalia aponta onde procurar:** no nível 1, a fusão não deu nada ao D linha (135,9 → 136,1), enquanto deu +2,7% ao P e +3,1% ao D exato. No nível 2 ela deu +8,2% ao linha. Os kernels fundidos não usam memória local (sem spill no `cuobjdump -res-usage`), mas o D1 linha fundido usa 61 registradores contra 56 do P1. O kernel K = 1 linha soma um terceiro `dp4a` (o do bit r) e lê dois inteiros de cada tabela, o que pode tirar ocupação. Só um perfil na GPU (Nsight Compute) separa as duas coisas.
- **Limite desta medida:** os bits são os do Lloyd (o mais perto), não os do GPTQ. A régua de qualidade do PR #15 passou com o D exato e bits GPTQ; com tabela linha e bits GPTQ, ainda não foi medido.

Próximos passos: (1) medir no compressor o D linha com bits GPTQ contra a régua do PR #15; (2) aqui, um perfil do kernel D1 linha fundido para fechar os 4% que faltam contra o Q5_K_M no nível 1.

### Perfil do kernel (A100-40GB, sem modelo, 2026-10-27)

[`tests/kernel/strata_kernel_d_perfil_colab.ipynb`](../tests/kernel/strata_kernel_d_perfil_colab.ipynb) mede o MMVQ de uma coluna nas formas do FFN do 8B (4096 → 14336) com blocos aleatórios; resultado em [`resultados/kernel_d_perfil_a100.md`](../tests/kernel/resultados/kernel_d_perfil_a100.md).

| kernel | só o up (µs) | gate+up fundido (µs) | registradores (fundido) | warps ativos (fundido) | instruções (fundido) |
|---|---:|---:|---:|---:|---:|
| Q5_K | 48,4 | 95,0 | 62 | 43% | 2,59 × 10⁷ |
| P1 | 49,7 | 93,7 | 56 | 49% | 2,85 × 10⁷ |
| D1 exato | 62,3 | 112,7 | 56 | 50% | 3,86 × 10⁷ |
| D1 linha | 51,6 | 101,0 | **61** | **43%** | 2,97 × 10⁷ |

A causa da anomalia é a ocupação. Sozinho, o D1 linha fica 4% atrás do P1, a mesma diferença das instruções. Fundido, ele usa 61 registradores, e o A100 cabe só 8 blocos de 128 threads por SM, contra 9 do P1 com 56. Os warps ativos caem de 49% para 43%, e o kernel fica 8% atrás do P1. Analogia: é a mesma esteira com uma vaga a menos para operários.

Correção (commit seguinte): `strata_min_blocos` pede ao compilador 9 blocos por SM (`__launch_bounds__`) só no MMVQ fundido de uma coluna dos tipos de bloco Q4_K com tabela (D, D linha e livro). Compilado para sm_80, todos ficam em ≤ 56 registradores, sem pilha e com as mesmas instruções no laço (D1 linha: 192). Nos derivados do Q6_K, o limite faria o compilador guardar registradores na pilha, então eles ficam como estavam. O Q5_K do ggml também perde ocupação na fusão (62 registradores), mas ele é a referência e não se mexe.

Conferido no mesmo notebook (A100-40GB, commit `9756817`, [`resultados/kernel_d_perfil_a100_limite56.md`](../tests/kernel/resultados/kernel_d_perfil_a100_limite56.md)): o D1 linha fundido caiu de 101,0 para **95,6 µs** (P1: 93,6), com 56 registradores, 9 blocos por SM e 49% de warps ativos, iguais aos do P1. A fusão passou a dar +10,6% a ele (antes +4,3%). O D2 exato fundido também melhorou (133,3 → 127,5 µs); o D2 linha e o D1 exato ficaram onde estavam, porque já usavam ≤ 56. Os kernels sem fusão não mudaram (mesmo SASS). Uma medida destoa: o Q6_K_D1 linha só com o up deu 96,4 µs, contra 83,7 na rodada anterior e com o mesmo SASS; é ruído da medida ou o host não escolheu o kernel linha nesse caso, e a rodada do 8B vai mostrar qual.

### 8B com o limite de registradores (Colab A100-80GB, 2026-10-29)

Commit `9756817`, mesmo notebook, mesmos GGUFs regravados ([`resultados/unsloth__Meta-Llama-3.1-8B_kernel_d_a100_limite56.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_kernel_d_a100_limite56.md)). As KLs saíram idênticas às da primeira rodada.

| tok/s (tg128) | nível 1 | nível 2 |
|---|---:|---:|
| Q5_K_M / Q6_K imatrix | 146,2 | 119,7 |
| P | 141,3 | 145,9 |
| D exato | 123,4 | 121,6 |
| D linha | **137,9** | 140,6 |

| candidato | N1 / P | N1 / Q5_K_M | N2 / P | N2 / Q6_K | veredito |
|---|---:|---:|---:|---:|---|
| D exato | 0,87 | 0,84 | 0,83 | 1,02 | reprova |
| D linha | **0,98** | **0,94** | 0,96 | 1,17 | **zona cinza** |

- **A correção fez o que o perfil prometia:** a fusão agora dá +2,4% ao D linha no nível 1 (era +0,1%; o P ganha +2,6%). O D linha subiu de 0,96 para 0,98 do P e de 0,92 para 0,94 do Q5_K_M. Ficou a 0,01 da régua.
- **O que falta já não é do D:** o D linha está a 0,976 do P, e o próprio P está a 0,967 do Q5_K_M no nível 1. Para o D linha chegar a 0,95 do Q5_K_M, o P teria que ficar mais rápido. O nível 2 oscilou entre as sessões (P 147,3 → 145,9; D linha 0,97 → 0,96 do P), dentro da variação de uma máquina do Colab para outra.
- **Veredito pela régua:** D exato reprova; D linha fica na zona cinza, a 0,01 de passar.
- **O que decide se isso importa:** a thread do compressor achou que o "D com GPTQ passa" do PR #15 usava outra tabela. No 1B, com a tabela do kernel e bits GPTQ, o D linha empata com o P com GPTQ. Se o 8B confirmar (`strata_8b_gptq_linha_colab.ipynb`), o P com GPTQ é o formato e a velocidade do D deixa de importar. Se o D linha ganhar do P com GPTQ, ele já roda a 0,96–0,98 do P.

### D linha com bits GPTQ no 8B (thread do compressor, 2026-10-30)

Na thread do compressor (L4, KL em 200 trechos, `tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_gptq_dl_l4.md` no branch do PR #15), o D linha com bits GPTQ empatou com o P com GPTQ no nível 1 (KL ×1000 6,45 contra 6,49 no wikitext). No nível 2 ele ganhou nos dois textos: 2,81 contra 2,92 no wikitext e 4,15 contra 4,35 no C4. Pela régua do compressor, o resultado ficou na zona cinza (retenção de 36% no C4).

Para o kernel, a configuração que sai disso (nível 1 com pontos P e nível 2 com tabela linha) não pede nada novo. O GGUF do nível 2 usa os tipos D2 com tabela linha, e a tabela de 4 entradas por tensor representa qualquer par de pontos de nível 1 com os filhos de nível 2. O nível 2 linha passou na régua de velocidade deste documento: 0,96 do P e 1,17 do Q6_K. O nível 1 roda nos tipos P, que são a referência.

