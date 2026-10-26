# STRATA, livro de códigos aninhado no formato real (Qwen/Qwen3-8B-Base)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. KL contra o BF16 nos 100 primeiros trechos do wikitext-2 raw de teste; imatrix com 125 trechos do texto de treino. Bits por peso nas matrizes das camadas.

| modelo | bits/peso | KL (×1000) | mesmo 1º token | erro ponderado |
|---|---:|---:|---:|---:|
| F16_Q3_K_M_imat | 3.846 | 74.471 ± 0.836 | 86.22% | 0.6391% |
| livro_N0 | 4.250 | 31.130 ± 0.361 | 90.53% | 0.2250% |
| LIVRO0_gpu | 4.250 | 31.130 ± 0.361 | 90.53% | n/d% |
| F16_IQ4_XS_imat | 4.277 | 30.923 ± 0.325 | 91.01% | 0.2436% |
| F16_Q4_K_S_imat | 4.531 | 27.633 ± 0.307 | 91.13% | 0.1779% |
| F16_Q4_K_M_imat | 4.791 | 25.215 ± 0.261 | 91.19% | 0.1441% |
| livro_N1 | 5.250 | 16.726 ± 0.168 | 92.38% | 0.0995% |
| LIVRO1_gpu | 5.250 | 16.726 ± 0.168 | 92.38% | n/d% |
| F16_Q5_K_S_imat | 5.500 | 11.990 ± 0.102 | 93.44% | 0.0484% |
| F16_Q5_K_M_imat | 5.650 | 12.167 ± 0.100 | 93.34% | 0.0530% |
| F16_Q6_K_imat | 6.562 | 9.226 ± 0.062 | 93.85% | 0.0298% |
| F16_Q6_K_Q8mix_imat | 7.489 | 7.486 ± 0.045 | 94.35% | 0.0115% |
| F16_Q8_0 | 8.500 | 3.013 ± 0.019 | 96.61% | 0.0080% |

## Régua de qualidade (curva = k-quants com imatrix em F16)

| nível | bits/peso | KL ×1000 | curva no mesmo tamanho | razão | bits a mais | bits a mais (curva com IQ4_XS) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.250 | 31.130 | 41.511 | 0.750 | -0.199 | -0.024 |
| 1 | 5.250 | 16.726 | 15.585 | 1.073 | +0.067 | +0.067 |

**Veredito de qualidade (régua de docs/09): zona cinza** (níveis 0/1: -0.024 / +0.067 bit/peso)

Llama-3.1-8B (livro em float, 2026-10-23): +0,004 / +0,006.

**Dominância** (menos bits e menos KL ao mesmo tempo, sem interpolação):

- nível 0: nenhum
- nível 1: nenhum

## Conferência do kernel

| modelo | caminho F16 | GPU, lote normal | diferença / erro padrão | GPU, lote 8 | CPU (5 trechos) | GPU lote 8 (5 trechos) | CPU − GPU lote 8 / erro padrão |
|---|---:|---:|---:|---:|---:|---:|---:|
| nível 0 | 31.130 | 31.130 | +0.00 | 31.551 | 51.063 | 35.634 | +9.44 |
| nível 1 | 16.726 | 16.726 | +0.00 | 17.103 | 32.607 | 19.420 | +16.61 |
| IQ4_XS_imat (controle) | 30.923 | | | 31.497 | 51.688 | 36.932 | +9.02 |
| Q5_K_S_imat (controle) | 11.990 | | | 12.542 | 28.761 | 13.895 | +26.74 |

**Conferência: NÃO passou** (as duas diferenças dos níveis a no máximo 1 erro padrão). KL ×1000. A coluna "GPU, lote 8" mostra o efeito de arredondar as ativações para 8 bits na geração; ele deve ser do mesmo tamanho nos tipos do livro e nos de controle.

## Velocidade na GPU (tok/s, mediana de rodadas alternadas)

| modelo | geração (tg128) | prompt (pp512) |
|---|---:|---:|
| LIVRO1 | 134.7 | 6732 |
| LIVRO0 | 167.9 | 6836 |
| IQ4_XS_imat | 181.7 | 4715 |
| Q5_K_S_imat | 144.8 | 4446 |
| Q4_K_M_imat | 146.6 | 4536 |

- geração, nível 0 / IQ4_XS: 0.924 (régua ≥ 0,95: NÃO passou)
- geração, nível 1 / Q5_K_S: 0.930 (régua ≥ 0,95: NÃO passou)
- prompt, nível 0 / IQ4_XS: 1.450; nível 1 / Q5_K_S: 1.514 (fora da régua: os tipos do livro leem o prompt por desquantização + cuBLAS; os do llama.cpp, pelo MMQ)
