# Livro de códigos aninhado: o estrato 0 também vira livro

Plano escrito em 2026-10-23, antes de rodar qualquer medida. Pergunta: se o estrato 0 deixar de ser a grade uniforme do Q4_K_M e passar a ser um livro de códigos não uniforme, ajustado com a imatrix, e cada bit a mais dividir cada valor do livro em dois (a ideia do Any-Precision LLM, [pesquisa](07-quantizacao-aninhada.md)), o preço de aninhar some?

## O que já foi tentado e não se repete aqui

Quatro vertentes mexeram só nos níveis de cima e deixaram o nível 0 como o Q4_K_M ([resultados, seções 28 e 30](05-resultados.md#30-sub-células-por-tensor-no-llama-31-8b-tipos-d-na-gpu)). A melhor delas foi a dos tipos D, com deslocamentos de Lloyd por tensor sobre a grade intacta do Q4_K_M. No Llama-3.1-8B, ela ficou em +0,23 bit/peso no nível 1 e +0,11 no nível 2 contra os k-quants com imatrix no mesmo tamanho (KL ×1000 9,27 e 3,40 contra 7,13 e 3,09 interpolados). O que é novo aqui:

- **O nível 0 também é livro.** O livro não herda as 16 marcas uniformes do Q4_K.
- **Aninhamento por divisão.** O código do nível k é 2 × código do nível k − 1 + 1 bit, e o filho pode cair em qualquer ponto dentro da célula do pai.
- **Custo contado em bits por peso,** já com escalas e tabelas.
- **O estrato 0 deixa de ser igual ao Q4_K_M do Hub,** e esse custo vai para a conta (ver abaixo).

## Os dois desenhos

[`tests/kernel/strata_livro.py`](../tests/kernel/strata_livro.py) grava cada nível como um GGUF com as matrizes das camadas em F16. Assim a qualidade é medida pelo `llama-perplexity` sem kernel novo. Todo ajuste minimiza Σ imatrix_j · (x_ij − q_ij)², o mesmo critério do `llama-quantize`.

| desenho | o que guarda | bits/peso na RAM (nível de b bits) |
|---|---|---|
| **linha** (Any-Precision puro) | um livro de 2^b valores F16 por linha da matriz, k-means ponderado 1D; cada nível divide cada célula em duas com 2-médias dentro dela | b + 16·2^b / colunas (8B, 4096 colunas: 4,06 / 5,13 / 6,25) |
| **bloco** (híbrido estilo IQ4_XS) | uma escala por bloco de 32 pesos (6 bits com sinal × um F16 por 256, a conta do IQ4_XS) e um livro por tensor, ajustado com escalas e livro em rodadas alternadas, partindo da tabela do IQ4_NL; os filhos dividem cada valor do livro, por tensor ou por linha | b + 0,25 (+ tabela por linha, se os filhos forem por linha) |

Para separar o preço de aninhar do resto, `--independente` ajusta cada largura do zero, sem aninhar.

Todas as matrizes das camadas usam a mesma largura. O Q4_K_M dá 6 bits a algumas matrizes sensíveis, então essa escolha é conservadora: se o livro empatar sem a mistura, a mistura só ajuda.

## Régua de decisão (definida antes de rodar)

Eixo x: bits por peso nas matrizes das camadas, contando escalas e tabelas (para os k-quants, os bytes reais do GGUF). Eixo y: KL ×1000 contra o BF16. Os dois lados ficam no caminho F16, porque o livro não tem kernel e o F16 não arredonda as ativações (seção 28). A curva é a dos k-quants com imatrix (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0), interpolada pelo log da KL. O IQ4_XS com imatrix entra como ponto à parte: ele já é "escala por bloco + tabela de 16 valores" e está no Hub.

**No 1B (container), só para descartar:**

- Descarta se o melhor desenho, no mesmo tamanho, gastar mais de +0,15 bit/peso que a curva em qualquer nível.
- Descarta também se a soma dos três níveis não ficar abaixo da dos tipos D medidos do mesmo jeito.
- Passar no 1B não prova nada: o 1B já prometeu mais do que o 8B entregou quatro vezes.

**No 8B (Colab A100), quem decide:**

| resultado | condição | o que acontece |
|---|---|---|
| **sucesso** | os níveis 0, 1 e 2 a no máximo +0,05 bit/peso da fronteira (k-quants e IQ4_XS) | o preço de aninhar some; vale abrir o kernel |
| **ganho forte** | sucesso, e além disso algum nível ≥ 0,10 bit/peso abaixo da fronteira | idem, e vira o caminho principal |
| **zona cinza** | algum nível entre +0,05 e +0,10 | registra e não abre kernel; os tipos D continuam |
| **fracasso** | algum nível acima de +0,10, ou soma ≥ 0,34 (a dos tipos D) | descarta |

**Ajuste feito depois do 1B e antes do 8B,** e só para deixar a régua mais exigente:

- **O que mudou:** a fronteira ganhou pontos onde a curva tinha buracos: o Q4_K_S, o Q5_K_S e um Q6_K com as matrizes de atenção e a `ffn_down` em Q8_0 (7,24 bits/peso no 1B). O IQ4_XS passou a fazer parte da fronteira em todos os níveis, não só como condição do nível 0.
- **Por quê:** no 1B, a interpolação pelo log da KL entre o Q3_K_M e o Q4_K_M (1 bit de distância) dava ao nível 0 uma vantagem de −0,36 bit/peso. Entre o Q6_K e o Q8_0 (2 bits), dava −0,38 ao nível 2. Contra o IQ4_XS, a vantagem do nível 0 some (−0,01). Isso era do buraco na curva, não do formato.
- **O que vale independente da curva:** a **dominância**, menos bits e menos KL ao mesmo tempo. O notebook lista as dominâncias à parte.

Por que o empate é o mínimo: com os tipos D, o nível 0 é o Q4_K_M do Hub e roda em qualquer llama.cpp. O livro abre mão disso, então só compensa se tirar todo o preço dos níveis de cima.

## O custo de o estrato 0 não ser o Q4_K_M do Hub

- **Quem já baixou o Q4_K_M do Hub não reaproveita nada.** A conta da seção 31 deixa de valer: hoje, quem tem o Q4_K_M baixa 1,00 GB para o nível 1 e 6,25 GB para o BF16 exato. Com o livro, baixa o estrato 0 de novo (~4,3 GB no 8B).
- **O nível 0 não roda no llama.cpp sem patch.** Hoje roda. Todo nível precisaria de tipo novo no ggml, inclusive o mais baixo, que é o que roda nas máquinas pequenas.
- **"Um arquivo no lugar de BF16 + Q4 + Q5 + Q6" continua valendo** se o fecho até o BF16 custar o mesmo (~3,3 bits/peso), mas o arquivo deixa de poder ser montado "ao lado" de um GGUF publicado (especificação B.10).

## Kernel: o que ele compra, separado da qualidade

O kernel não muda a KL: a KL é do formato. Ele compra duas coisas:

1. **Rodar o formato.** Sem kernel, o livro só existe na emulação em F16.
2. **Velocidade.** Na geração, o gargalo é ler os pesos da memória. Então a meta é empatar com o k-quant do mesmo tamanho (régua de sempre: ≥ 0,95), não ganhar.

A RAM é a dos bits por peso da tabela acima. O kernel não tira nada disso.

Dá para medir uma parte já, sem escrever kernel: o nível 0 do desenho bloco tem o mesmo formato do IQ4_XS (escala de 6 bits por 32, códigos de 4 bits, tabela de 16 valores int8 consultada com `pshufb`), só que a tabela vem do tensor em vez de ser fixa. Então a velocidade do IQ4_XS no `llama-bench` é a velocidade do nível 0. Os níveis 1 e 2 consultam tabelas de 32 e 64 valores: duas ou quatro consultas de 16 por grupo de pesos, ou `vpermb` no AVX-512 VBMI. Esse custo a mais é o que os tipos D já mostraram na GPU (0,65× dos tipos P) e precisa de micro-benchmark antes de qualquer promessa.

Sobre o kernel "próprio, fora do llama.cpp" como caminho para democratizar: um kernel fora do ggml perde Ollama, LM Studio e os backends Metal, Vulkan e CUDA que o llama.cpp já tem. Democratizar é qualidade por byte boa *e* rodar onde as pessoas já estão. O caminho, se a qualidade passar, é um tipo novo dentro do ggml, como os tipos P e D.

## Resultados

### Triagem no Llama-3.2-1B (contêiner)

Caminho F16, KL contra o BF16 nos 40 primeiros trechos do wikitext-2 de teste, imatrix do texto de treino ([tabela completa e JSON](../tests/kernel/resultados/unsloth__Llama-3.2-1B_livro_container.md)). Bits por peso nas matrizes das camadas; erro ponderado = Σ imatrix·(x − q)² / Σ imatrix·x².

| modelo | bits/peso | KL ×1000 | mesmo 1º token | erro ponderado |
|---|---:|---:|---:|---:|
| Q3_K_M | 3,842 | 127,89 ± 2,03 | 82,5% | 1,851% |
| livro por linha, nível 0 | 4,099 | 174,58 ± 2,67 | 79,4% | 1,303% |
| **livro por bloco, nível 0** | 4,250 | 49,46 ± 1,08 | 89,0% | 0,472% |
| IQ4_XS | 4,272 | 47,68 ± 0,73 | 89,0% | 0,507% |
| Q4_K_S | 4,539 | 43,05 | | 0,428% |
| Q4_K_M | 4,802 | 38,76 ± 0,63 | 90,0% | 0,420% |
| livro por linha, nível 1 | 5,198 | 43,33 ± 0,73 | 89,5% | 0,324% |
| **livro por bloco, nível 1** | 5,250 | 17,36 ± 0,34 | 93,0% | 0,157% |
| Q5_K_S | 5,500 | 11,53 | | 0,110% |
| Q5_K_M | 5,656 | 11,79 ± 0,22 | 94,4% | 0,109% |
| **livro por bloco, nível 2** | 6,250 | 4,98 ± 0,08 | 96,0% | 0,042% |
| livro por linha, nível 2 | 6,397 | 10,89 ± 0,25 | 94,7% | 0,074% |
| Q6_K | 6,562 | 5,37 ± 0,12 | 95,9% | 0,028% |
| Q6_K com atenção e `ffn_down` em Q8_0 | 7,431 | 2,17 | 97,2% | 0,018% |
| Q8_0 | 8,500 | 0,62 ± 0,02 | 98,4% | 0,003% |

Bits a mais que a fronteira no mesmo KL (abaixo de 0 = o livro gasta menos):

| candidato | nível 0 | nível 1 | nível 2 | soma |
|---|---:|---:|---:|---:|
| curva só com Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 (a do plano original) | −0,356 | −0,128 | −0,380 | |
| **livro por bloco, fronteira da régua** | **−0,006** | **−0,014** | **−0,384** | **−0,40** |
| livro por linha, fronteira da régua | +0,393 | +0,676 | +0,817 | +1,89 |
| tipos D por tensor (seção 27, 5,80 e 6,80 bits/peso), fronteira da régua | 0 | +0,373 | −0,021 | +0,35 |
| livro por bloco **sem aninhar** (`--independente`, cada largura ajustada do zero; não é STRATA) | −0,006 | −0,200 | −0,585 | |

O que isso diz:

- **O livro por linha (Any-Precision puro) está descartado.** O nível 0 fica pior que o Q3_K_M gastando mais bits. Um livro por linha sem escala por bloco não acompanha a variação de tamanho dos pesos ao longo da linha: o erro ponderado é 2 a 3 vezes o do livro por bloco em todos os níveis.
- **O livro por bloco passou no filtro do 1B.** Nenhum nível passa de +0,15 bit/peso, e a soma (−0,40) fica abaixo da dos tipos D (+0,35).
- **Os pontos novos da curva mudaram a conta, como se temia.** O nível 1 parecia 0,13 bit/peso à frente; com o Q5_K_S na fronteira, virou empate (−0,01). O nível 0 já era empate com o IQ4_XS. Os −0,38 do nível 2 ainda vêm de uma interpolação entre o Q6_K e o Q6_K com Q8_0 (0,87 bit de distância), então podem encolher do mesmo jeito.
- **O que não depende de interpolação:** o nível 2 domina o Q6_K (6,25 contra 6,56 bits/peso, KL 4,98 contra 5,37). Só isso já é pelo menos 0,31 bit/peso a menos no mesmo KL.
- **O preço de aninhar continua lá, ~0,19 bit/peso por nível.** Ajustado do zero, o mesmo formato de 5,25 bits/peso dá KL 12,58 (contra 17,36 aninhado) e fica 0,20 bit/peso à frente da fronteira. Com 6,25 bits/peso, dá 4,04 contra 4,98. Ou seja, o livro melhorou o formato em ~0,2 bit/peso, e o aninhamento devolveu quase tudo no nível 1. É da mesma ordem dos +0,11 a +0,23 que as vertentes anteriores pagavam contra os k-quants no 8B; o livro só começou de um ponto melhor.
- **Filhos por linha não resolvem no 1B:** a tabela por linha custa 16·2^b/colunas, 0,25 bit/peso no nível 1 com 2048 colunas, mais que o preço que ela tentaria tirar. No 8B (4096 colunas) custa 0,125 e 0,25; o notebook tem a variante `bloco_linha`, desligada por padrão.
- **Em resumo, no 1B:** empate nos níveis 0 e 1, ganho no nível 2. Pela régua do 8B isso seria "ganho forte". Mas é o padrão que já falhou no 8B (seções 26 a 30: o 1B prometia no nível 2 e o 8B não entregou). Quem decide é o notebook.

### Kernel no contêiner (CPU, AVX2)

`llama-bench` do llama.cpp sem patch, Llama-3.2-1B, tok/s:

| | geração 1 thread | geração 4 threads | prompt 1 thread | prompt 4 threads |
|---|---:|---:|---:|---:|
| Q4_K_M | 6,74 | 25,54 | 76,6 | 327 |
| IQ4_XS | 6,95 | 24,51 | 108,9 | 370 |
| IQ4_XS / Q4_K_M | 1,03 | 0,96 | 1,42 | 1,13 |

O nível 0 do livro por bloco tem o formato do IQ4_XS (só a tabela de 16 valores muda), então a velocidade dele é essa: a tabela de 16 sai de graça.

Micro-benchmark do produto escalar ([`tests/kernel/bench_livro.c`](../tests/kernel/bench_livro.c), conferido contra uma referência escalar), 16384 × 4096 pesos, ns por peso, 1 thread, melhor de várias rodadas (o contêiner é ruidoso, ±20%):

| | bits/peso | ns/peso | em relação ao nível 0 |
|---|---:|---:|---:|
| Q4_K (ggml) | 4,50 | 0,039 | |
| IQ4_XS (ggml) | 4,25 | 0,046–0,053 | |
| Q5_K (ggml) | 5,50 | 0,044–0,052 | |
| Q6_K (ggml) | 6,56 | 0,033–0,045 | |
| livro nível 0 (tabela de 16, `pshufb`) | 4,25 | 0,071–0,080 | 1 |
| livro nível 1 (tabela de 32: 2 `pshufb` + `blendv`) | 5,25 | 0,087–0,101 | ~1,2–1,3 |
| livro nível 2 (tabela de 64: 4 `pshufb` + 3 `blendv`) | 6,25 | 0,115–0,134 | ~1,5–1,7 |

- **Meu nível 0 ainda é ~1,5× mais lento que o IQ4_XS do ggml,** com o mesmo formato. Falta otimização. Então os números absolutos não valem; vale a razão entre os níveis.
- **Tabelas de 32 e 64 custam ~1,2× e ~1,5× no cálculo (com o ruído, até 1,3× e 1,7×).** Na geração, o gargalo costuma ser ler a memória, e o custo a mais se esconde em parte. Na GPU, os tipos D (que também consultam tabela) ficaram em 0,65× dos tipos P (seção 30). Essa é a estimativa honesta até existir kernel de verdade.
- **O processador do contêiner tem AVX-512 VBMI** (`vpermb` consulta 64 bytes de uma vez). Lá a tabela de 64 custaria uma instrução, mas a maioria das máquinas de consumo não tem.

### Llama-3.1-8B (Colab A100): sucesso pela régua, mas é empate

[Notebook](../tests/kernel/strata_8b_livro_colab.ipynb) rodado por Cantyll em 2026-10-23 numa A100-SXM4-80GB. KL contra o BF16 nos 100 primeiros trechos do wikitext-2 raw de teste, imatrix com 125 trechos do texto de treino, caminho F16 nos dois lados. Tabela completa em [`resultados/unsloth__Meta-Llama-3.1-8B_livro_a100.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_livro_a100.md) e JSON ao lado.

| modelo | bits/peso | KL ×1000 | mesmo 1º token | erro ponderado |
|---|---:|---:|---:|---:|
| Q3_K_M | 3,844 | 76,62 ± 1,20 | 87,7% | 1,678% |
| **livro, nível 0** | 4,250 | 33,52 ± 0,58 | 91,7% | 0,439% |
| IQ4_XS | 4,274 | 31,66 ± 0,53 | 91,8% | 0,467% |
| Q4_K_S | 4,536 | 28,96 ± 0,51 | 92,0% | 0,395% |
| Q4_K_M | 4,797 | 25,32 ± 0,49 | 92,7% | 0,385% |
| **livro, nível 1** | 5,250 | 12,38 ± 0,21 | 94,8% | 0,147% |
| livro sem aninhar, 5 bits | 5,250 | 9,32 ± 0,19 | 95,3% | 0,111% |
| Q5_K_S | 5,500 | 8,21 ± 0,16 | 95,7% | 0,101% |
| Q5_K_M | 5,653 | 7,55 ± 0,14 | 96,0% | 0,099% |
| **livro, nível 2** | 6,250 | 4,28 ± 0,07 | 96,9% | 0,039% |
| livro sem aninhar, 6 bits | 6,250 | 3,61 ± 0,06 | 97,1% | 0,032% |
| Q6_K | 6,562 | 3,31 ± 0,06 | 97,1% | 0,026% |
| Q6_K com atenção e `ffn_down` em Q8_0 | 7,457 | 1,57 ± 0,02 | 98,0% | 0,017% |
| Q8_0 | 8,500 | 0,57 ± 0,01 | 99,0% | 0,003% |

Bits a mais que a fronteira da régua (k-quants com imatrix + IQ4_XS), conferidos de novo com as mesmas funções da triagem:

| | nível 0 | nível 1 | nível 2 | soma | veredito |
|---|---:|---:|---:|---:|---|
| **livro aninhado (STRATA)** | **+0,004** | **+0,006** | **−0,027** | **−0,017** | **sucesso** |
| livro sem aninhar (não é STRATA, só mede o preço) | +0,004 | −0,171 | −0,216 | −0,383 | (ganho forte) |
| tipos D sobre o Q4_K_M do Hub (seção 30) | 0 | +0,23 | +0,11 | +0,34 | fracasso |
| 1B, livro aninhado (triagem acima) | −0,006 | −0,014 | −0,384 | −0,40 | |

Velocidade na A100, llama.cpp sem patch: o IQ4_XS gera a 195 tok/s contra 155 do Q4_K_M (1,26×), e o prompt fica igual (1,02×). O nível 0 do livro tem o mesmo formato.

O que isso diz:

- **É a primeira vez que os níveis do STRATA empatam com os k-quants na mesma RAM no 8B.** Os tipos D pagavam +0,23 e +0,11 bit/peso nos níveis 1 e 2. O livro tirou esse preço.
- **É empate, não ganho.** Nenhum nível domina um k-quant. Os níveis 1 e 2 caem entre pontos da curva separados por 0,70 bit (Q4_K_M a Q5_K_S) e 0,91 bit (Q5_K_M a Q6_K). Se a curva real é convexa no log da KL, essa interpolação favorece o livro em alguns centésimos. O ±0,05 da régua cobre isso, mas não sobra margem para chamar de vantagem.
- **O preço de aninhar continua ~0,18–0,20 bit/peso por nível.** Sem aninhar, o mesmo formato ficaria 0,17 e 0,22 à frente. O livro melhora o formato e o aninhamento devolve quase tudo, tanto no 1B quanto no 8B.
- **O 1B acertou os níveis 0 e 1 desta vez e exagerou o nível 2** (−0,38 no 1B, −0,03 no 8B). A regra de só concluir no 8B continua valendo.
- **Eu tinha previsto zona cinza ou fracasso** a partir do custo de aninhar no 1B. A previsão estava errada: o 1B já mostrava empate nos níveis 0 e 1, e eu descontei demais por causa das vezes em que ele exagerou.

O que a régua manda e o que ainda falta antes do kernel:

- Pela régua, sucesso abre o kernel. Mas o preço registrado acima continua de pé: o estrato 0 deixa de ser o Q4_K_M do Hub e até o nível 0 precisa de llama.cpp modificado.
- **Fecho até o BF16 exato:** ninguém mediu quanto custa partindo do livro. É ele que decide se "um arquivo no lugar de BF16 + Q4 + Q5 + Q6" continua de pé.
- **Tabelas de 32 e 64 na GPU:** na CPU custam ~1,2× e ~1,5×. Na GPU, os tipos D (que também consultam tabela) ficaram em 0,65× dos tipos P.

## Opção A: fecho e GPU antes do kernel (decisão de Cantyll, 2026-10-23)

Antes de escrever o kernel, duas perguntas: quanto custa o arquivo sem perdas partindo do livro, e quanto as tabelas de 32 e 64 valores custam na GPU.

### Fecho até o BF16 exato (Llama-3.2-1B, contêiner)

`strata_livro.py --modo bloco --bits 4 5 6 7 --fecho` ([JSON](../tests/kernel/resultados/unsloth__Llama-3.2-1B_livro_fecho_container.json)). Funciona como a estimativa da seção 15 de `05-resultados.md`:

- Quem leu o nível k sabe em que célula do livro o peso está: entre os pontos médios no nível 0, e do lado do corte que decidiu o filho nos níveis seguintes.
- O fecho escolhe o BF16 dentro da célula ao custo de −log2 p(w) + log2 P(célula).
- `p` é uma tabela de sinal+expoente por tensor e por oitava da escala do bloco, com mantissa uniforme.

Os níveis são planos crus de 1 bit, e os cortes F16 de cada nível estão contados. Nenhum peso caiu fora da sua célula (0 escapes em 973 M).

| lendo até | bits do nível | fecho | **sem perdas** |
|---|---:|---:|---:|
| nível 0 | 4,25 | 6,67 | 10,92 |
| nível 1 | 5,25 | 5,64 | 10,89 |
| nível 2 | 6,25 | 4,59 | 10,84 |
| nível 3 | 7,25 | 3,57 | **10,82** |

Mesmas matrizes, outras bases:

| arquivo sem perdas | bits/peso |
|---|---:|
| 0.2, sinal+expoente (seção 15) | 10,59 |
| **livro com 3 níveis, estimado** | **10,82** |
| 0.3, Q4_K_M com 3 níveis, gravado (seção 21) | 11,12 |
| 0.3, Q4_K_M, estimado só do nível 0 (seção 15) | 11,24 |

- **O livro é ~0,3 bit/peso mais barato que o Q4_K_M como base sem perdas.** Pela mesma conta de entropia, lendo só do nível 0, fica em 10,92 contra 11,24. Com 3 níveis, fica em 10,82 (estimado) contra 11,12 (gravado). A diferença vem das escalas: o livro gasta 0,25 bit/peso com elas (uma por bloco de 32), contra 0,5 do Q4_K (escala e mínimo por sub-bloco). Acima do 0.2, sobram +0,23, quase só essas escalas.
- **"Um arquivo no lugar de BF16 + Q4 + Q5 + Q6" continua de pé, e fica menor.** Os níveis empatam com os k-quants (8B, acima), e o arquivo inteiro sai ~3% menor que o 0.3 de hoje.
- **Ressalva:** é estimativa de entropia, não arquivo gravado. No 0.3, a estimativa do fecho (≈3,30) bateu com o gravado (3,32); aqui o modelo de contexto é o mesmo, trocando a oitava do passo pela oitava da escala do bloco. Medido no 1B; no 8B, a seção 31 mostrou o 0.3 com o mesmo número de bits por peso (11,13).

### Tabelas na GPU (micro-benchmark CUDA, Tesla T4)

[`bench_livro_gpu.cu`](../tests/kernel/bench_livro_gpu.cu) ([notebook](../tests/kernel/strata_livro_gpu_colab.ipynb), [resultado](../tests/kernel/resultados/livro_gpu_t4.md)). Produto matriz-vetor em 8 matrizes 14336 × 4096 (a `ffn` do 8B), ativação em int8, `__dp4a`. Cada formato do livro (tabela int8 nos registradores, consultada com `__byte_perm` como o IQ4_NL do llama.cpp) é comparado com uma decodificação uniforme que lê os mesmos bytes. Conferido contra a referência na CPU.

| | bits/peso | µs por matriz | GB/s lidos | tabela contra uniforme dos mesmos bytes |
|---|---:|---:|---:|---:|
| nível 0, tabela de 16 | 4,31 | 173,6 | 182 | 0,96× |
| nível 1, tabela de 32 | 5,31 | 199,0 | 196 | 0,96× |
| nível 2, tabela de 64 | 6,31 | 291,4 | 159 | **0,78×** |

(cópia na T4: 243 GB/s, leitura + escrita; o uniforme de 4 bits lê a 190 GB/s.)

- **As tabelas de 16 e 32 quase não custam:** −4%, escondidos atrás da leitura da memória.
- **A de 64 custa 22%.** São 15 `__byte_perm` para cada 4 pesos (4 consultas de 16 e 3 seleções), e o kernel deixa de ser limitado pela memória. Comparado com o Q6_K, que lê 4% a mais, o nível 2 deve ficar perto de 0,8× dele. É melhor que os 0,65× dos tipos D, mas ainda é custo.
- **A T4 representa bem as GPUs de consumo** (razão entre conta e banda parecida com a de uma RTX). Numa A100 a memória é bem mais rápida em relação à conta, e o custo das tabelas deve aparecer mais. O notebook roda igual lá.
- **O que daria para tentar no nível 2:** tabela em memória compartilhada, uma cópia por banco, com uma leitura por peso em vez de 15 seleções para cada 4 pesos. Isso precisa de medida, não de promessa.


## Kernel dos níveis 0 e 1 e segundo 8B (decisão de Cantyll, 2026-10-23)

Decisão: kernel do livro nos níveis 0 e 1 (o nível 2 fica para depois) e o teste de qualidade repetido em outro modelo de 8B. Régua escrita aqui antes de rodar.

**Formato real.** O nível 0 é o bloco do IQ4_XS byte a byte (136 bytes por 256 pesos: d em F16, 8 escalas de 6 bits, códigos de 4 bits), com a tabela de 16 valores int8 vinda do tensor em vez da tabela fixa do IQ4_NL. O nível 1 é esse bloco mais um plano de 32 bytes (168 bytes, 5,25 bits/peso), com tabela de 32 valores int8. As tabelas vão no GGUF por tensor e, na carga, para o `op_params` do tensor, como as dos tipos D. Uma escala α por tensor leva o livro para int8, a mesma nos dois níveis, porque o bloco do nível 0 é compartilhado. Diferente do Llama-3.1-8B, que usou o livro em float, aqui a medida inclui o arredondamento das tabelas para int8.

**Modelo:** `Qwen/Qwen3-8B-Base` (Apache 2.0, sem login; outra família, 36 camadas, `ffn` de 12288). Mesmo protocolo do Llama-3.1-8B: KL contra o BF16 nos 100 primeiros trechos do wikitext-2 raw de teste, imatrix com 125 trechos do texto de treino, A100.

**Qualidade (quem decide), caminho F16 nos dois lados** (k-quants desquantizados e livro desquantizado do formato real), na fronteira k-quants com imatrix (Q3_K_M, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q6_K+Q8_0, Q8_0) mais IQ4_XS:

| resultado | condição |
|---|---|
| **sucesso** | níveis 0 e 1 a no máximo +0,05 bit/peso da fronteira |
| **zona cinza** | algum nível entre +0,05 e +0,10 |
| **fracasso** | algum nível acima de +0,10 |

Dominância conta à parte, como antes.

**Kernel (separado da qualidade):**

- **Conferência:** a KL do tipo novo, lido pelo kernel na GPU, tem de ficar a 1 erro padrão da KL da emulação em F16; a CPU tem de dar o mesmo que a GPU nos trechos conferidos.
- **Velocidade de geração na GPU** (`llama-bench`, tg128): nível 0 ≥ 0,95× o IQ4_XS, e nível 1 ≥ 0,95× o Q5_K_S, o k-quant mais próximo em tamanho acima dele.
