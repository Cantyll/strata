# Quantização aninhada: o que já existe e as três vertentes para o STRATA

Pesquisa feita em 2026-10-19, depois da base fina no Llama-3.1-8B (resultados, seção 25). Pergunta: como outros trabalhos lidam com o conflito entre o nível de baixo e os níveis de cima num modelo aninhado, e o que dá para trazer para blocos compatíveis com os k-quants do llama.cpp. Os números dos trabalhos citados vêm dos próprios artigos (links no fim); o que é dedução nossa está marcado como tal.

## Onde o STRATA está

KL ×1000 contra o BF16, Llama-3.1-8B, tudo com imatrix:

| nível | k-quant | STRATA sobre o Q4_K_M | base fina T=1 | base fina T=2 |
|---|---:|---:|---:|---:|
| 0 | Q4_K_M 26,3 | 26,3 | 33,1 | 34,2 |
| 1 | Q5_K_M 8,60 | 10,78 | 7,59 | 8,69 |
| 2 | Q6_K 3,47 | 3,66 | | 3,16 |

## O que os outros fizeram

| trabalho | como aninha | pós-treino? | quem paga o aninhamento | kernel |
|---|---|---|---|---|
| Any-Precision LLM (ICML 2024) | centróides não uniformes por canal (SqueezeLLM de 3 bits); cada nível divide cada grupo em dois com k-means ponderado | sim | quase ninguém: PPL 5,61 contra 5,62 no Llama-2-7B a 4 bits. Com grade uniforme (AWQ, GPTQ) o mesmo método quebra | planos de bits transpostos; GEMV 2,2 a 4× mais rápido que FP16; a transposição come 35 a 58% do tempo |
| Matryoshka Quantization (DeepMind, 2025) | int2 e int4 são os bits altos de um int8 | treino (QAT ou OmniQuant) | o nível de cima perde ~0,5 ponto, o int2 ganha 3 a 7 | não mediu |
| MatGPTQ (2026) | igual ao MatQuant, mas em uma passada de GPTQ com o erro médio dos níveis | sim | Llama-3.1-8B: 3 bits ganham muito (PPL 11,3 → 9,1), 6 e 8 bits perdem (6,27 → 6,57) | tensor cores, 2,6× sobre FP16 a 4 bits |
| AnyBCQ (NAVER, 2026) | planos binários com escala própria por nível | sim, com reconstrução por bloco | ~0,7 a 1,1 ponto de MMLU em todo nível, mesmo com escala própria | LUT sem transposição, 186 tok/s a 4 bits |
| RRQ (2026) | base de poucos bits + estágios residuais, cada um com escala própria | sim, sem calibração | bom em 6 e 8 bits; arquivo ~5% maior | não tem |

- **Ninguém resolveu o caso do STRATA.** Nenhum trabalho mantém um k-quant com imatrix intacto como base e ainda tem níveis de cima de graça. Todo esquema com grade uniforme tem a mesma troca que medimos: ou a base é ótima e os níveis de cima perdem, ou o contrário.
- **A exceção é a grade não uniforme** (Any-Precision): os filhos podem ficar em qualquer ponto dentro da célula do pai, e aí o aninhamento quase não custa. É a pista mais nova para nós.
- **Escala própria por nível não fecha a diferença** (AnyBCQ), o que bate com a nossa seção 26.
- **Rodar GPTQ para escolher os arredondamentos** rende pouco nos k-quants: o AutoRound no Q4_K_S do Llama-3.1-8B leva a média de 0,6228 a 0,6239 e demora 20× mais. Fica de fora.
- **Teoria:** fontes gaussianas com erro quadrático podem ser refinadas aos poucos sem perder nada (Equitz e Cover, 1991). Isso vale para o quantizador ótimo, não para uma grade uniforme com escala fixa por bloco, que é o nosso caso (dedução).

## Kernel

- O Q5_K e o Q6_K do llama.cpp já guardam os bits altos num plano separado (`qh`). O STRATA faz o mesmo com bits baixos, e o custo de juntar é deslocar e somar, não transpor como no Any-Precision (dedução). O kernel não é o gargalo: a geração já está entre 0,86 (nível 2 na CPU) e 1,20 (nível 2 na GPU) do k-quant.
- Na GPU, o k-quant usa MMQ (tensor cores int8) na leitura do prompt e o STRATA usa desquantização + cuBLAS. Em lotes grandes o cuBLAS chega a ganhar do MMQ (0,77 a 0,96× no PR #7860 do llama.cpp), o que explica o prompt mais rápido do STRATA na A100. Um MMQ para os tipos P ficaria para depois.
- Vale manter os planos separados no arquivo e na memória mapeada (cada nível lê só o que precisa) e montar os blocos na carga só para o caminho tiled da CPU, como já está.

## As três vertentes

Cada uma representa uma prioridade diferente:

1. **Escala conjunta, Q4 quase intacto.** A busca de escala e mínimo do Q4_K passa a minimizar λ0·E0 + λ1·E1 + λ2·E2 (erro de cada nível, ponderado pela imatrix), com os códigos de 4 bits arredondados ao mais próximo e os níveis dividindo a célula como hoje. O nível 0 continua um Q4_K comum. Os tipos P não mudam, então não há kernel novo. Uma varredura de λ1/λ0 em {0; 0,25; 0,5; 1; 2} mostra a curva entre o Q4_K_M intacto (26,3 / 10,78) e algo perto da base fina.
   - **Pergunta:** existe um ponto em que o nível 1 fica abaixo de 8,60 (Q5_K_M) e o nível 0 custa só uns 5% (abaixo de ~27,6)?
2. **Base fina T=1 com nível 2.** É a que já ganha no nível 1 (7,59). Falta medir o nível 2 dela no 8B. O tamanho na RAM sai da conta do Q6_K empacotado.
   - **Pergunta:** o nível 2 empata com o Q6_K?
3. **Sub-células não uniformes, Q4_K_M intacto byte a byte.** Ideia tirada do Any-Precision: em cada tensor, o filho de cima e o de baixo de cada código q4 ganham deslocamentos próprios, ajustados com a imatrix (32 números por tensor por nível). O nível 0 é o próprio Q4_K_M do hub. Medida por emulação em F16, como na seção 26; o kernel precisaria de uma tabelinha por tensor.
   - **Pergunta:** isso pega o que a escala por bloco não pegou (8,81 contra 7,48 do Q5_K_M no caminho F16)?

A distribuição de níveis por tensor (dar planos só onde a KL mais cai por bit) entra depois, aplicada à vertente que ganhar.

## Régua de decisão (definida antes de rodar)

1. **Qualidade por byte:** KL de cada nível contra a curva dos k-quants com imatrix (Q4_K_M, Q5_K_M, Q6_K e Q8_0 medidos no mesmo notebook), no mesmo tamanho de RAM. Os níveis 1 e 2 pesam mais (decisão de 2026-10-19).
2. **Custo no nível 0:** quanto o nível 0 perde contra o Q4_K_M com imatrix, em bits por peso equivalentes.
3. **Velocidade:** pelo menos 0,95 do k-quant de mesma qualidade na geração. As vertentes 1 e 2 usam os tipos de hoje. A 3 só ganha kernel se vencer em qualidade.
4. **Desempate:** o nível 0 igual byte a byte ao Q4_K_M dos hubs.

## Resultado no 8B

Nenhuma vertente empata com os k-quants com imatrix no mesmo tamanho ([resultados, seção 28](05-resultados.md#28-três-vertentes-no-llama-31-8b-nenhuma-ganha-no-mesmo-tamanho)). Pela régua, a vertente 3 por tensor fica na frente: menor custo somado (+0,21 bit no nível 1, +0,10 no nível 2) e nível 0 igual ao Q4_K_M do hub.

## Fontes

- [Any-Precision LLM](https://arxiv.org/html/2402.10517v4) · [código](https://github.com/SNU-ARC/any-precision-llm)
- [Matryoshka Quantization](https://arxiv.org/html/2502.06786v3)
- [MatGPTQ](https://arxiv.org/html/2602.03537v1)
- [AnyBCQ](https://arxiv.org/html/2510.10467v2)
- [RRQ](https://arxiv.org/html/2608.04048)
- [Drop-by-Drop](https://arxiv.org/html/2606.12876) · [MoBiQuant](https://arxiv.org/html/2602.20191v2) · [BitStack](https://arxiv.org/html/2410.23918v3) · [QuIP#](https://arxiv.org/html/2402.04396v2)
- [Equitz e Cover, refinamento sucessivo](https://isl.stanford.edu/~cover/papers/transIT/0269equi.pdf)
- [AutoRound em GGUF](https://github.com/intel/auto-round/blob/main/docs/gguf_alg_ext_acc.md)
- llama.cpp: [k-quants, PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684) · [MMQ contra cuBLAS, PR #7860](https://github.com/ggml-org/llama.cpp/pull/7860)
