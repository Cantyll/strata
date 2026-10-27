# STRATA, C1: bits dos níveis escolhidos com a Hessiana (Llama-3.1-8B, caminho F16)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. KL contra o BF16 em 100 trechos de 512 tokens de cada texto (wikitext-2 raw de teste; C4 en, validação). imatrix e Hessianas com 500 trechos da calibração misturada (wikitext de treino + C4 validação 00001); amortecimento 0.1. Matrizes das camadas: 6.98 bilhões de pesos. Nível 0 = Q4_K_M com imatrix (4.583 GiB); RAM do nível 1 5.395 GiB, do nível 2 6.208 GiB.

## KL (×1000)

| modelo | GiB | wikitext | C4 | mesmo 1º token (wiki) |
|---|---:|---:|---:|---:|
| Q4_K_M_imat | 4.583 | 25.38 ± 0.49 | 26.22 ± 1.09 | 92.66% |
| Q5_K_S_imat | 5.215 | 8.47 ± 0.21 | 7.61 ± 0.23 | 95.72% |
| Q5_K_M_imat | 5.339 | 7.67 ± 0.16 | 7.57 ± 0.28 | 96.06% |
| Q6_K_imat | 6.143 | 3.30 ± 0.06 | 3.22 ± 0.11 | 97.17% |
| Q8_0 | 7.954 | 0.57 ± 0.01 | 0.55 ± 0.03 | 98.96% |
| P_perto_N1 | 5.395 | 10.51 ± 0.18 | 9.89 ± 0.32 | 95.20% |
| P_perto_N2 | 6.208 | 3.54 ± 0.06 | 3.40 ± 0.09 | 97.07% |
| P_gptq_N1 | 5.395 | 7.00 ± 0.15 | 7.42 ± 0.30 | 96.11% |
| P_gptq_N2 | 6.208 | 3.19 ± 0.07 | 3.25 ± 0.17 | 97.41% |
| D_gptq_N1 | 5.395 | 6.83 ± 0.13 | 7.15 ± 0.27 | 96.16% |
| D_gptq_N2 | 6.208 | 3.01 ± 0.05 | 3.08 ± 0.15 | 97.34% |

## Régua (bits/peso a mais que a frente dos k-quants em F16, no mesmo tamanho)

| candidato | nível | wikitext | C4 | ganho sobre o P de hoje mantido no C4 |
|---|---:|---:|---:|---:|
| P_perto | 1 | +0.375 | +0.387 |  |
| P_perto | 2 | +0.162 | +0.141 |  |
| P_gptq | 1 | -0.038 | +0.046 | 71% |
| P_gptq | 2 | +0.037 | +0.088 | 44% |
| D_gptq | 1 | -0.066 | +0.002 | 76% |
| D_gptq | 2 | -0.038 | +0.021 | 61% |

**Veredito pela régua:** ZONA CINZA: ver a tabela (algum nível entre +0,05 e +0,12, só os D chegam lá, ou o ganho não se mantém no C4).

## Erro de saída das camadas (Σ ΔW H ΔWᵀ na calibração, relativo ao do nível 0)

| candidato | nível 1 | nível 2 |
|---|---:|---:|
| P_perto | 0.4388 | 0.1053 |
| P_gptq | 0.2352 | 0.0728 |
| D_gptq | 0.2332 | 0.0678 |

Tempo total: 257 min.
