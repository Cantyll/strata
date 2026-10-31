# Quantização abaixo de 4 bits e codecs de pesos: o que já foi medido e onde estão as lacunas

Pesquisa feita em 2026-10-31. Pergunta: o que a literatura e os repositórios de inferência local já mediram sobre bits por peso, qualidade e custo de decodificação, como isso se cruza com os números do STRATA, e onde há espaço que ninguém explorou.

Regras deste documento:

- Os números dos trabalhos citados vêm das próprias fontes (links no fim). Onde a fonte é um gráfico, a leitura é nossa e está marcada como "leitura do gráfico".
- O que é conta ou conclusão nossa está marcado como **dedução**.
- Uma medida é nossa: o experimento N2 (trellis contra k-quants e i-quants na CPU, Llama-3.2-1B), rodado no contêiner depois da autorização do Cantyll. Resultado na seção 10 e em [`tests/trellis/resultados/`](../tests/trellis/resultados/Llama-3.2-1B-Instruct_n2_cpu.md). O resto vem das fontes.

## Resumo

1. **A régua do STRATA não é a fronteira de qualidade por bit.** No Llama-3.1-8B Instruct, o EXL3 (variante do QTIP, com trellis) a 4,0 bits tem KL 0,013, menor que o Q4_K_M com imatrix (0,018 a ~4,8 bits). No mesmo KL, o trellis gasta ~1,0 bit a menos no tamanho do Q4_K_M, ~1,2 no do Q5_K_M e ~1,5 no do Q6_K (leitura do gráfico + dedução, seção 4). O STRATA passou semanas disputando ±0,1 bit contra os k-quants com imatrix; essa curva está ~1 bit atrás do melhor que já roda em placa de vídeo de consumo.
2. **A teoria prevê esse buraco.** Numa fonte gaussiana a 2 bits, o quantizador escalar ótimo fica 0,46 bit acima do limite teórico, o livro E8P do QuIP# 0,25 bit e o trellis do QTIP 0,07 bit (seção 5). Grade escalar com escala por bloco é o que os k-quants e os níveis do STRATA usam.
3. **Essa fronteira é só de GPU NVIDIA.** EXL3 exige CUDA 12.4; QTIP só mediu GPU. Na CPU e no Apple, o que existe é o llama.cpp (k-quants, IQ) e o fork ik_llama.cpp, que tem os tipos trellis `IQ*_KT`. Medimos no 1B (N2, seção 10): na CPU o trellis do ik_llama.cpp ganha só 0,16 a 0,23 bit da curva do llama.cpp oficial no mesmo KL, e o IQ4_KT gera texto a 0,75× do Q4_K_M. **O ~1 bit do EXL3 na GPU não apareceu na CPU** (com a ressalva de que é 1B contra 8B). O que mais ganhou na CPU foi o IQ4_K do mesmo fork, que não é trellis: 0,29 bit menor que o Q4_K_M, com KL menor e 1,10× mais rápido.
4. **Sem perda está saturado.** ZipNN 66,4%, DFloat11 ~68–70%, STRATA 0.2 66,2% do BF16. A folga que sobra é de ~0,07 bit (medida nossa, seção 6).
5. **Menos bits não quer dizer mais rápido.** O IQ2_XXS é mais lento que o Q4_0 no Metal; o DFloat11 é ~2× mais lento que o BF16 em lote 1; o AQLM 1×16 só ganha 1,2–1,3× do FP16 na GPU (seção 7).
6. **A lacuna mais específica para o STRATA:** trellis aninhado. A teoria do refinamento sucessivo de trellis existe desde 1999 (Jafarkhani e Tarokh) e o Any-Precision mostrou que aninhar um quantizador não uniforme quase não custa, mas não achamos nenhum trabalho que aninhe trellis em pesos de LLM (seção 8, G1).

## 1. Como ler os números sem se enganar

Quatro armadilhas aparecem em quase toda comparação entre artigos:

- **Contexto da perplexidade.** O QTIP mede o Llama 2 com contexto 4096 e o Llama 3 com 8192; o VPTQ mede 3 bits com 2048; o llama.cpp mede com 512. PPL com contextos diferentes não se comparam diretamente. Comparamos razões (PPL quantizado / PPL original) ou KL do mesmo protocolo.
- **Família do modelo.** O Llama 3 sofre mais que o Llama 2 no mesmo formato. Na tabela do llama.cpp, o Q4_K_M sem imatrix tem KL 0,0127 no Llama-2-7B e 0,0313 no Llama-3-8B; o Q2_K tem razão de PPL 1,108 contra 1,565. Resultados de 2 bits no Llama 2 não se transferem para o Llama 3.
- **O que conta como bit.** Os artigos de 2 bits contam só as camadas quantizadas: "Bits count only quantized weights" (AQLM). No Llama-3-8B, embedding + saída têm 2 × 128.256 × 4.096 = 1,05 bilhão de parâmetros, 13% de 8,03 bilhões. Se ficam em FP16, um modelo "de 2 bits" ocupa (2 × 6,98 + 16 × 1,05) / 8,03 ≈ **3,8 bits por peso no total** (dedução). O gráfico do EXL3 também conta só o decoder. O STRATA conta "bits das matrizes de camada", que é comparável a isso.
- **Ajuste fino depois de quantizar.** QuIP#, AQLM, QTIP (código HYB) e PV-Tuning usam ajuste fino. O QuIP# sem ajuste vai de 6,19 para 8,22 no Llama-2-7B a 2 bits. Os k-quants e o STRATA não ajustam nada.

## 2. Qualidade por bit: Llama 2 (onde os artigos se comparam entre si)

Wikitext-2, contexto 4096, com ajuste fino. FP16: 7B 5,12; 70B 3,12. Fontes: QTIP tabela 5, AQLM, PV-Tuning, QuIP#.

| bits | método | 7B | 70B |
|---|---|---:|---:|
| 4 | AQLM | 5,21 | 3,19 |
| 4 | QuIP# | 5,19 | 3,18 |
| 4 | QTIP (HYB) | 5,17 | 3,16 |
| 3 | AQLM | 5,38 | 3,36 |
| 3 | QuIP# | 5,41 | 3,35 |
| 3 | QTIP (HYB) | 5,28 | 3,26 |
| 2 | AQLM | 6,14 | 3,83 |
| 2 | QuIP# | 6,19 | 3,91 |
| 2 | QTIP (HYB) | 5,86 | 3,70 |
| 2,02 | PV-Tuning (AQLM + ajuste) | 5,84 | 3,78 (2,07 b) |
| 1,02 | PV-Tuning | 8,28 | 5,52 (1,14 b) |

Outros pontos de referência:

- **VPTQ** (Llama-2-7B, 2,02 b): 6,13 no Wikitext-2 e 8,07 no C4, contra 6,19 / 8,16 do QuIP#. Quantizar o 70B custou 19 horas de GPU, contra 183 do AQLM e ~25 do QuIP#.
- **IQ2_XXS do llama.cpp** (2,06 b, livro de 256 pontos da rede E8, sem ajuste fino): Llama-2-7B 7,067 e 70B 4,079 com contexto 4096, segundo o PR #4773. Fica atrás dos métodos com ajuste fino (6,19 e 3,91 do QuIP#).
- **HQQ** (sem calibração, menos de 5 minutos no 70B): Llama-2-70B a 2 bits com grupo 16 dá PPL 4,12, contra 4,63 do Llama-2-13B em FP16 (blog do HQQ, via busca; não abrimos a página).
- **EntQuant** (2026, sem dados, código de entropia ANS sobre Float8 otimizado para baixa entropia): Llama-2-70B a 3 bits C4 5,74 contra 5,85 do GPTQ; a 2,1 bits 6,47 contra 6,12 do QuIP#. Comprime um 70B em menos de 30 minutos numa H100. Roda 1,5–2× mais devagar que o BF16.

**Leitura:** a 4 bits todos empatam com o original perto de 1–2%. A diferença entre os métodos aparece a 3 bits e explode a 2 bits. O trellis (QTIP) lidera de 2 a 4 bits; o PV-Tuning empata com ele a 2 bits graças ao ajuste fino.

## 3. Llama 3 8B (o modelo que importa para o STRATA)

**Artigos (Wikitext-2, contexto 8192, BF16 = 5,54):**

| bits (só decoder) | QuIP# | QTIP | PV-Tuning |
|---|---:|---:|---:|
| 4 | 5,81 | 5,67 | |
| 3 | 6,27 | 6,01 | |
| 2 | 7,84 | 7,33 | 6,99 (2,01 b) |
| 1 | | | 11,17 (1,01 b) |

**llama.cpp (README do `llama-perplexity`, Llama-3-8B, contexto 512, KL contra FP16, imatrix "WT 10m").** Bits por peso no total = tamanho em GiB × 8 × 2³⁰ / 8,03 × 10⁹ (dedução):

| tipo | GiB | b/peso total | PPL (FP16 6,233) | KL |
|---|---:|---:|---:|---:|
| Q6_K | 6,14 | 6,57 | 6,253 | 0,0055 |
| Q5_K_M | 5,33 | 5,70 | 6,289 | 0,0108 |
| Q4_K_M imatrix | 4,58 | 4,90 | 6,383 | 0,0282 |
| IQ4_XS imatrix | 4,14 | 4,43 | 6,460 | 0,0363 |
| Q3_K_M imatrix | 3,74 | 4,00 | 6,734 | 0,0844 |
| IQ3_M | 3,53 | 3,78 | 6,898 | 0,1025 |
| IQ3_XXS | 3,05 | 3,26 | 7,458 | 0,1836 |
| IQ2_M | 2,74 | 2,93 | 8,601 | 0,3260 |
| IQ2_XXS | 2,24 | 2,40 | 14,09 | 0,8120 |
| IQ1_S | 1,88 | 2,01 | 58,1 | 2,21 |

Os contextos diferentes impedem comparar PPL entre as duas tabelas. A razão de PPL do QTIP a 4 bits (5,67 / 5,54 = 1,023) é a mesma do Q4_K_M com imatrix (6,383 / 6,233 = 1,024), mas o QTIP conta 4 bits só no decoder; com embedding e saída em FP16 ele ocuparia ~5,6 bits no total, contra 4,9 do Q4_K_M (dedução). **Comparar artigos com GGUF pelo tamanho do arquivo pode inverter o resultado.** Por isso a seção 4 usa uma única fonte, com um único protocolo.

## 4. A comparação de um protocolo só: o gráfico do EXL3

O repositório do exllamav3 publica um gráfico de KL contra bits por peso (só decoder) no Llama 3.1 8B Instruct, com EXL3, EXL2, GGUF com imatrix, AQLM e VPTQ medidos pelo mesmo autor ([`doc/llama31_8b_instruct_kld_bpw.png`](https://github.com/turboderp-org/exllamav3/tree/master/doc)). O README diz que o EXL3 "é uma variante simplificada do QTIP". O texto usado e o contexto não estão no README. Valores lidos do gráfico (posições em bits com erro de ±0,05):

| formato | b/peso | KL |
|---|---:|---:|
| EXL3 2,0 | 2,0 | 0,312 |
| EXL3 2,5 | 2,5 | 0,151 |
| EXL3 3,0 | 3,0 | 0,053 |
| EXL3 3,5 | 3,5 | 0,030 |
| EXL3 4,0 | 4,0 | 0,013 |
| EXL3 5,0 | 5,0 | 0,003 |
| EXL3 6,0 | 6,0 | 0,001 |
| AQLM 2 bits 1×16 + PV | ~2,27 | 0,243 |
| AQLM 2 bits 2×8 + PV | ~2,0 | 0,308 |
| VPTQ v8-k65536-4096 | ~3,8 | 0,052 |
| GGUF IQ2_M | ~2,7 | 0,245 |
| GGUF IQ3_XS | ~3,27 | 0,092 |
| GGUF IQ3_M | ~3,57 | 0,070 |
| GGUF IQ4_XS | ~4,27 | 0,023 |
| GGUF Q4_K_M | ~4,8 | 0,018 |
| GGUF Q5_K_M | ~5,65 | 0,007 |
| GGUF Q6_K | ~6,56 | 0,003 |
| EXL2 4,0 | 4,0 | 0,054 |

**Mesmo KL, quantos bits o EXL3 precisa** (interpolação em log(KL) entre os pontos vizinhos do EXL3; dedução):

| GGUF com imatrix | bits GGUF | bits EXL3 no mesmo KL | diferença |
|---|---:|---:|---:|
| IQ2_M | 2,70 | 2,16 | −0,54 |
| IQ3_XS | 3,27 | 2,74 | −0,53 |
| IQ4_XS | 4,27 | 3,66 | −0,61 |
| Q4_K_M | 4,80 | 3,80 | −1,00 |
| Q5_K_M | 5,65 | 4,42 | −1,23 |
| Q6_K | 6,56 | 5,00 | −1,56 |

A diferença cresce com os bits. Isso bate com a teoria da seção 5: o erro de uma grade escalar fixa se afasta do limite à medida que a taxa sobe, e o trellis não.

**Onde o STRATA cai nesse gráfico** (dedução, comparando cada nível com os k-quants que o próprio STRATA mediu no Llama-3.1-8B base, [resultados, seção 32](05-resultados.md#32-livro-de-códigos-aninhado-no-llama-31-8b-empate-nos-três-níveis) e [docs/10](10-motor-de-compressao.md)):

| nível do STRATA | bits | KL ×1000 | distância da curva k-quant | distância estimada do EXL3 |
|---|---:|---:|---:|---:|
| livro, nível 0 | 4,25 | 33,52 | +0,004 b | ~+0,6 b |
| livro, nível 1 | 5,25 | 12,38 | +0,006 b | ~+1,0 a +1,2 b |
| livro, nível 2 | 6,25 | 4,28 | −0,027 b | ~+1,3 a +1,5 b |
| P / D linha com GPTQ, nível 2 | ~6,5 | 2,81 (wikitext) | dentro de +0,05 b | ~+1,5 b |

O protocolo é diferente (base contra Instruct, texto e contexto diferentes), então só a ordem de grandeza vale. Mas ela é clara: **o melhor nível do STRATA empata com os k-quants, e os k-quants ficam de 0,5 a 1,5 bit atrás do trellis.** No 8B, 1 bit por peso no decoder são ~0,87 GB de RAM.

## 5. Por que existe esse buraco: a conta da teoria

O QTIP mediu o erro quadrático (MSE) de quantizadores numa fonte gaussiana i.i.d. a 2 bits (tabela 1 do artigo). Convertemos para bits acima do limite teórico com Δ = ½·log₂(MSE / D(R)), onde D(2) = 2⁻⁴ = 0,0625 (dedução):

| quantizador | dimensão | MSE | bits acima do limite |
|---|---:|---:|---:|
| escalar Lloyd-Max | 1 | 0,118 | 0,46 |
| QuIP# E8P (rede E8) | 8 | 0,089 | 0,25 |
| QTIP trellis (1MAD, 3INST) | 256 | 0,069 | 0,07 |
| limite teórico | ∞ | 0,063 | 0 |

Em taxa alta, o resultado clássico é que o melhor quantizador escalar de taxa fixa fica ~0,72 bit acima do limite numa gaussiana (fórmula de Panter e Dite), e um escalar uniforme seguido de codificador de entropia fica 0,25 bit acima (Gish e Pierce). Os k-quants são escalares, uniformes e de taxa fixa por bloco. O trellis chega a 0,07 bit do limite com um código que se decodifica em 2 a 4 instruções por peso na GPU (QTIP).

Duas consequências para o STRATA (dedução):

- **Quantização escalar (k-quants, níveis do STRATA, livro) tem um teto.** Melhorar escalas, bits GPTQ, mistura por tensor e tabela não uniforme mexe em décimos de bit. A teoria deixa ~0,4–0,7 bit na mesa que só uma quantização vetorial ou de trellis recupera. Isso explica por que o motor de compressão parou na zona cinza.
- **Rotação de Hadamard sozinha deve render pouco nos k-quants** (as notas do motor de compressão registram ≈0,04 b, mas a medida não está nos documentos do repositório), porque a escala por bloco já absorve os pesos grandes. No QTIP e no QuIP#, a rotação serve para deixar os pesos parecidos com uma gaussiana, que é o que o trellis e a rede E8 precisam. O ganho vem da combinação.

## 6. Codecs sem perda

| codec | o que comprime | tamanho | decodificação | onde roda |
|---|---|---|---|---|
| ZipNN (IBM) | expoente separado + Huffman | BF16 66,4% (Llama-3.1); FP32 ~83%; modelos "limpos" até ~45% | 1,65 GB/s por núcleo (M1 Max); até 80 GB/s com 16 processos | CPU, armazenamento |
| DFloat11 | Huffman no expoente do BF16 | 67,9–70%, ~10,8–11,1 b; Llama 3.1 405B de 811,7 para 551,2 GB | na GPU, por bloco; lote 1 ~2× mais lento que o BF16; 2,3–46× mais rápido que descarregar para a CPU | só GPU CUDA |
| ECF8 | expoente de modelos FP8 | economiza 9,8–14,8% em LLMs FP8 e até 26,9% em modelos de vídeo | GPU, custo do kernel não isolado | GPU |
| nvCOMP ANS | genérico | 79% (comparação no artigo do DFloat11) | DFloat11 até 21× mais rápido | GPU |
| STRATA 0.2 | expoente com rANS + planos | 66,2%, 10,59 b (Llama-3.2-1B) | leitor Rust/JS na CPU | CPU, armazenamento |
| STRATA 0.3 | Q4_K_M do Hub + níveis + fecho | 11,12 b, volta byte a byte ao BF16 e ao Q4_K_M | CPU | armazenamento |

O que cruza dessas fontes:

- **No BF16, todo mundo chega no mesmo lugar** (~10,6–11 b), porque todos atacam só o expoente. O STRATA mediu que a folga restante, com contexto entre expoente e mantissa, é ~0,07 b ([resultados, seção 12](05-resultados.md)). Ganhar do DFloat11 em tamanho é questão de centésimos.
- **Arquivo já quantizado quase não comprime.** O ZipNN diz que GGUF "não comprime nada" e GPTQ/AWQ ficam em 85–91%. O STRATA mediu 3,86 bits de entropia nos índices de 4 bits do Q4_K (ganho de ~0,12 b). O EntroLLM (2025) diz economizar até 65% contra uint4, mas com a quantização dele, cuja grade desperdiça códigos (dedução: um bom quantizador de taxa fixa usa os códigos de forma quase uniforme).
- **Onde codificador de entropia rende de verdade é quando ele escolhe a quantização**, não depois dela: é o caso do EntQuant (taxa variável, sem dados). O preço é a decodificação na hora de usar (1,5–2× mais lento que o BF16 na GPU).

## 7. Custo do kernel: o que cada formato paga na hora de gerar texto

| formato | hardware | medida | fonte |
|---|---|---|---|
| QTIP 2 / 3 / 4 b | RTX 6000 Ada, Llama-2-7B, lote 1 | 188 / 161 / 140 tok/s; FP16 55,9; QuIP# 2 b 186; AQLM 2 b 81,5 | QTIP tab. 4 |
| QuIP# 2 b | RTX 4090, Llama-2-7B | 106,3 tok/s (HF); FP16 33,1; AQLM 20,6; com FlashAttention 170,5 | QuIP# tab. 5–6 |
| AQLM 1×16 / 2×8 | RTX 3090, matvec | 1,2–1,31× / 1,57–3,05× o FP16 | AQLM |
| AQLM 2×8 | CPU i9 8 núcleos | 2,75–3,69× o FP32 | AQLM |
| IQ2_XXS contra Q4_0 | 7B, geração | CUDA 4080: 155 contra 130; Metal M2 Max: **54 contra 63,5**; CPU 5975WX: 24,3 contra 15,4 tok/s | llama.cpp PR #4773 |
| IQ2_KT (trellis) | RTX 4080, Llama-3.1-8B | 146–194 tok/s contra 208 do IQ2_XXS | ik_llama.cpp PR #113 |
| Any-Precision 3 / 4 / 8 b | RTX 4090, GEMV 4096² | 3,99 / 3,03 / 1,56× o cuBLAS FP16 | Any-Precision tab. 6 |
| any4 | GPU, lotes 1–16 | até 2× o BF16 (int4 até 3×) | any4 |
| DFloat11 | GPU, lote 1 | ~2× mais lento que o BF16 | README do DFloat11 |
| EntQuant | GPU | 1,5–2× mais lento que o BF16 | EntQuant |
| BitNet b1.58 2B4T | CPU, bitnet.cpp | 29 ms/token, 0,4 GB sem embedding (contra 41–124 ms de modelos FP16 comparados) | relatório técnico |
| STRATA tipos P | GPU / CPU | 1,00× o Q5_K_M na GPU; 1,03–1,11× na CPU com 1 thread | resultados, kernel |
| STRATA livro | GPU A100 | 0,924× o IQ4_XS | docs/09 |

O que cruza dessas fontes (dedução):

- **Geração em lote 1 é limitada por memória, mas só até certo ponto.** O QTIP a 2 bits lê 8× menos bytes que o FP16 e gera 3,4× mais rápido, não 8×. Abaixo de ~3 bits, o custo de decodificar passa a pesar.
- **Livro grande mata a velocidade.** O AQLM 1×16 (livro de 2¹⁶ entradas) quase não ganha do FP16 na GPU. Livros que cabem no cache (E8P de 1 KiB, IQ2 de 256 entradas, tabelas de 16–32 do STRATA) ou códigos calculados (1MAD, 3INST) não pagam isso.
- **O mesmo formato muda de lado conforme o hardware.** O IQ2_XXS ganha do Q4_0 na CPU e na CUDA, mas perde no Metal. Para inferência local, a comparação precisa ser feita em cada backend.
- **Custo de quantizar** também separa os métodos: AQLM 183 horas de GPU no 70B, VPTQ 19, QuIP# ~25, EXL3 "algumas horas" numa GPU de consumo, HQQ menos de 5 minutos, EntQuant menos de 30. Para o Cantyll (T4/Kaggle), só os de baixo cabem.

## 8. LACUNAS

Cada lacuna diz o que foi procurado e não achado. "Não achamos" quer dizer isso: a busca pode ter deixado passar algo.

**G1. Trellis aninhado para pesos de LLM.**
- O que existe: refinamento sucessivo de trellis em teoria e em imagem (Jafarkhani e Tarokh, IEEE Trans. Inf. Theory 1999; patente US 6.125.149; Wang e Orchard 2003). A gaussiana é refinável sem perda no limite (Equitz e Cover, 1991). O Any-Precision aninha um quantizador escalar não uniforme de 3 a 8 bits quase de graça (Llama-2-7B a 4 bits: 5,62 aninhado contra 5,61 separado). O STRATA mediu que aninhar o livro custa ~0,18 b por nível no 8B.
- O que não achamos: nenhum trabalho que aninhe trellis (QTIP, EXL3, IQ*_KT) ou rede E8 em pesos de LLM. Os trabalhos de aninhamento (Any-Precision, MatQuant, MatGPTQ, AnyBCQ, Bit-by-Bit) usam grades escalares ou planos binários.
- Por que importa: é a interseção exata entre o que o STRATA tem de próprio (um arquivo com vários tamanhos) e a maior folga medida na literatura (~1 bit). Se o aninhamento custar pouco num trellis, os níveis do STRATA saem da disputa de ±0,1 b e vão para a frente da curva. Ressalva do N2: na CPU, o trellis disponível hoje ganhou só ~0,2 b no 1B. O valor de G1 depende de a folga de ~1 bit do EXL3 se confirmar na plataforma que o STRATA escolher.
- Risco: o trellis do QTIP lê pesos em sequência dentro de um bloco; um nível de refinamento precisa de outro caminho no trellis ou de um sub-trellis por transição (o caminho da patente). Ninguém mediu o custo disso num LLM.

**G2. Trellis na CPU, medido de forma comparável.**
- O que existe: o ik_llama.cpp tem `IQ1_KT` a `IQ4_KT` e diz que o trellis inteiro foi feito para rodar razoavelmente na CPU. No PR original, o autor mediu IQ2_KT 28,7% de erro de quantização contra 33,2% do QTIP a 2 bits no Llama-2-7B, e disse que o trellis precisa de ~0,2 bit a menos que os tipos i- e iqk- no mesmo erro no Llama-3.1-8B.
- O que não achamos: uma tabela com o mesmo modelo, a mesma KL e tok/s na CPU (x86 e ARM) de IQ4_KT contra Q4_K_M e IQ4_XS. Os tipos KT não estão no llama.cpp oficial.
- Por que importa: o público de modelo local roda llama.cpp em CPU e Apple. O plano de MVP do STRATA mira um notebook x86 sem GPU. Se o trellis ganhar ~0,5–1 bit também na CPU sem perder velocidade, é a melhor alavanca de RAM disponível hoje.
- **Medido depois (N2, 1B):** não ganhou. 0,16–0,23 b no mesmo KL, IQ4_KT a 0,75× da velocidade do Q4_K_M. A lacuna continua aberta para o 8B e para ARM, mas a aposta ficou mais fraca.

**G3. Fecho sem perda a partir de uma base trellis ou vetorial.**
- O que existe: o STRATA fecha até o BF16 exato a partir do Q4_K_M (11,12 b) e estimou que a partir do livro sai ~0,3 b mais barato (1B).
- O que não achamos: ninguém mediu quantos bits custa voltar ao BF16 exato a partir de um modelo QTIP, EXL3, AQLM ou VPTQ.
- Obstáculo (dedução): esses formatos quantizam no espaço rotacionado (Hadamard). Para voltar byte a byte ao BF16, a rotação inversa precisa dar exatamente o mesmo resultado em ponto flutuante em qualquer máquina, ou o resíduo precisa ser guardado no espaço original. As duas coisas são possíveis, mas não são de graça.

**G4. Contabilidade honesta do embedding e da saída.**
- Os artigos de 2 bits contam só o decoder. No Llama-3-8B, embedding e saída são 13% dos parâmetros. Mantidos em FP16, levam um "2 bits" a ~3,8 bits no total; quantizados a 6 bits, a ~2,6 (dedução). O QTIP ainda relata que quantizar uma camada específica do Llama 3 70B (`v` da camada 0) derruba o modelo sem um truque de tokens.
- Não achamos estudo sistemático de quanto se pode quantizar embedding e saída abaixo de 4 bits sem estragar o modelo. Para modelos pequenos (1B–3B, os que rodam em máquina popular) essa fração é ainda maior.

**G5. Uma tabela única: mesmo modelo, mesma KL, todos os backends.**
- Os números de velocidade estão espalhados por GPUs diferentes (3090, 4080, 4090, RTX 6000 Ada, A100), modelos diferentes (Llama 2 7B, Llama 3.1 8B) e implementações diferentes (HF, FlashAttention, Triton sem otimização). Não achamos uma tabela que meça k-quants, IQ, KT, EXL3, AQLM e DFloat11 no mesmo modelo, com a mesma KL, em CPU x86, Apple e CUDA. É o tipo de cruzamento que o Cantyll pediu e que ninguém publicou.

**G6. Formato de guardar diferente do formato de rodar.**
- O levantamento "Great Inversion" (2026) argumenta que um codificador que distribui bits livremente prefere concentrar energia (transformada tipo KLT), enquanto o formato de taxa fixa que a GPU executa prefere espalhar (Hadamard). As duas metas apontam para lados opostos.
- O STRATA já guarda num formato e monta outro na carga (tipos P). Não achamos trabalho que use isso de propósito: guardar com taxa variável e transformada que concentra, e transcodificar na carga para um formato de taxa fixa. É a lacuna mais especulativa; o custo de transcodificar na carga não foi medido por ninguém.

## 9. O que o STRATA poderia adotar, em ordem

1. **Pôr uma segunda curva na régua.** Na GPU, o adversário de qualidade por bit é o EXL3 (trellis). Na CPU, pelo N2, é o IQ4_K / IQ3_K do ik_llama.cpp (0,16–0,32 b à frente do llama.cpp oficial no 1B), não o trellis. Sem uma dessas curvas, "empata com os k-quants" se lê como "empata com o estado da arte", e não é. Não verificamos se o EXL3 roda na T4.
2. **Escolher a plataforma de propósito.** Se o alvo é CPU e llama.cpp oficial (o plano de MVP diz notebook x86 de 8 GB), os k-quants são a régua prática certa, mas o documento deve dizer "fronteira prática do llama.cpp", não "fronteira". Se o alvo é qualidade por bit, o adversário é o trellis.
3. **Testar G1 primeiro no papel, sem modelo.** Experimento N1 abaixo. É barato, roda no contêiner e decide se vale abrir uma frente de trellis aninhado.
4. **Parar de investir em tamanho sem perda.** A folga é de ~0,07 b. O 0.2 e o 0.3 já estão no mesmo lugar que DFloat11 e ZipNN. O que diferencia o STRATA é o aninhamento com volta exata, não o tamanho.
5. **Relatar dois números de bits** em toda tabela: só das matrizes de camada (para comparar com artigos) e do arquivo inteiro (o que o usuário paga em RAM).
6. **Não adotar:** livros grandes estilo AQLM 1×16 (kernel lento e 183 horas de GPU no 70B); ajuste fino tipo PV-Tuning ou QAT (precisa de orçamento de treino que o projeto não tem); DFloat11 como formato de execução local (só GPU e 2× mais lento em lote 1).

## 10. Experimentos propostos (régua escrita antes)

**N1. Trellis aninhado numa gaussiana (contêiner, só numpy, sem modelo).**
- Mede o MSE em fonte gaussiana i.i.d., nas taxas 2→3→4 e 4→5→6 bits, de: (a) trellis independente por taxa (trellis de 8 estados do tipo Ungerboeck, mais simples que o do QTIP); (b) trellis aninhado por sub-trellis; (c) escalar aninhado como o STRATA faz hoje; (d) escalar Lloyd-Max independente; (e) o limite teórico.
- Régua: se o trellis aninhado custar **≤ 0,10 b por nível** contra o trellis independente, passa para o 1B. Se custar **> 0,20 b**, perde a vantagem de ~0,4 b sobre o escalar e a ideia morre. No meio, zona cinza.
- Custo estimado: 1 a 2 dias de trabalho no contêiner, sem Colab.

**N2. Trellis contra k-quant na CPU (contêiner, Llama-3.2-1B).**
- Compila o ik_llama.cpp, quantiza o 1B (BF16 + imatrix do bartowski) em Q4_K_M, IQ4_XS, IQ4_KT, IQ3_KT, Q5_K_M e Q6_K, mede KL contra o F16 no mesmo texto e tok/s com `llama-bench` em 4 threads.
- Régua: o IQ4_KT passa se ficar a ≥ 0,3 b da curva dos k-quants no mesmo KL e a ≥ 0,90× a velocidade de geração do Q4_K_M. Se passar no 1B, repetir no 8B (triagem no 1B não basta, regra do projeto).
- **Resultado (2026-10-31, contêiner, [tabela completa](../tests/trellis/resultados/Llama-3.2-1B-Instruct_n2_cpu.md)): reprova.** A curva de comparação acabou sendo a do llama.cpp oficial inteira (k-quants + i-quants, 19 tipos medidos), porque abaixo de 4 bits os i-quants são a frente. Embedding fixo em Q8_0 em todos.

| tipo | bits (decoder) | KL ×1000 | bits da curva oficial no mesmo KL | diferença | geração contra o vizinho de mesmo KL |
|---|---:|---:|---:|---:|---:|
| IQ2_KT (trellis) | 2,31 | 656 | 2,54 | −0,23 | 1,18× IQ2_M |
| IQ3_KT (trellis) | 3,18 | 164 | 3,38 | −0,20 | 1,43× IQ3_S |
| IQ4_KT (trellis) | 4,04 | 49,8 | 4,19 | −0,16 | **0,75× Q4_K_M** |
| IQ3_K (iqk, sem trellis) | 3,46 | 127 | 3,61 | −0,16 | 1,70× IQ3_S |
| IQ4_K (iqk, sem trellis) | 4,52 | 29,2 | 4,84 | −0,32 | 1,10× Q4_K_M |

  O IQ4_KT ganhou 0,16 b (régua: 0,3) e gerou a 0,75× do Q4_K_M (régua: 0,90). Pela regra, não vai para o 8B. Quantizar cada tipo trellis levou 4 a 10 minutos no 1B, contra ~30 s de um k-quant.
  Fora da régua: o IQ4_K, que não é trellis, domina o Q4_K_M (0,29 b menor, KL 29,2 contra 30,5, 1,10× mais rápido). Na CPU, a frente prática é o ik_llama.cpp com os tipos iqk, e ela fica de 0,1 a 0,3 b à frente da curva que o STRATA usa.
  Limites: 1B só; CPU com AVX-512 (notebook comum tem só AVX2); os k-quants rodaram nos kernels do fork, não nos do llama.cpp oficial.

**N3. Fecho a partir de um modelo trellis (depois de N2).**
- Mede a entropia do resíduo até o BF16 exato partindo do IQ4_KT, no espaço original. Só faz sentido se N2 passar. **N2 reprovou, então N3 fica parado** até haver um trellis que ganhe na plataforma alvo.

## Fontes

Quantização abaixo de 4 bits:
- [QTIP: Quantization with Trellises and Incoherence Processing (NeurIPS 2024)](https://arxiv.org/html/2406.11235)
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/html/2402.04396v2)
- [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/html/2401.06118)
- [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression](https://arxiv.org/html/2405.14852)
- [VPTQ: Extreme Low-bit Vector Post-Training Quantization](https://arxiv.org/html/2409.17066)
- [HQQ (repositório)](https://github.com/mobiusml/hqq) · [modelo 2 bits do Llama-2-70B](https://huggingface.co/mobiuslabsgmbh/Llama-2-70b-hf-2bit_g16_s128-HQQ)
- [HIGGS: Pushing the Limits of LLM Quantization via the Linearity Theorem (NAACL 2025)](https://arxiv.org/html/2411.17525v1)
- [YAQA: Model-Preserving Adaptive Rounding](https://arxiv.org/pdf/2505.22988)
- [any4: Learned 4-bit Numeric Representation for LLMs (ICML 2025)](https://arxiv.org/html/2507.04610v1)
- [ParetoQ (NeurIPS 2025)](https://arxiv.org/html/2502.02631v2)
- [BitNet b1.58 2B4T Technical Report](https://ar5iv.labs.arxiv.org/html/2504.12285)
- [Gemma 3 QAT (blog do Google)](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- [How Good Are Low-bit Quantized LLaMA3 Models?](https://arxiv.org/abs/2404.14047)
- [Any-Precision LLM (ICML 2024)](https://arxiv.org/html/2402.10517v4)
- [Transforms for LLM Quantization: The Great Inversion and Format Co-Design (2026)](https://arxiv.org/abs/2608.25188)
- [EntQuant: Float8@2bits](https://arxiv.org/html/2601.22787v1)
- [EntroLLM](https://arxiv.org/abs/2505.02380v1)

Repositórios de inferência local:
- [llama.cpp: tabela de KL do Llama 3 8B no README do `llama-perplexity`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- [llama.cpp PR #4773: IQ2_XXS](https://github.com/ggml-org/llama.cpp/pull/4773)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) · [PR #113: tipos trellis](https://github.com/ikawrakow/ik_llama.cpp/pull/113)
- [exllamav3 (EXL3)](https://github.com/turboderp-org/exllamav3) · [gráficos em `doc/`](https://github.com/turboderp-org/exllamav3/tree/master/doc)
- [Unsloth Dynamic 2.0 GGUFs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)
- [KL de quants GGUF no Mistral-7B (gist do Artefact2)](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)

Codecs sem perda:
- [ZipNN](https://arxiv.org/html/2411.05239v2)
- [DFloat11 (artigo)](https://arxiv.org/html/2504.11651) · [README](https://github.com/LeanModels/DFloat11)
- [ECF8: To Compress or Not?](https://arxiv.org/html/2510.02676v1)
- [Huff-LLM](https://arxiv.org/abs/2502.00922)

Teoria:
- [Jafarkhani e Tarokh, Design of Successively Refinable Trellis-Coded Quantizers (1999)](https://engineering.uci.edu/files/Jafarkhani-Design-of-Successively-Refinable-July-1999.pdf) · [patente US 6.125.149](https://patents.justia.com/patent/6125149)
- [Equitz e Cover, refinamento sucessivo (1991)](https://isl.stanford.edu/~cover/papers/transIT/0269equi.pdf)
- Panter e Dite (1951) e Gish e Pierce (1968): resultados clássicos de quantização em taxa alta (0,72 b e 0,25 b acima do limite numa gaussiana).
