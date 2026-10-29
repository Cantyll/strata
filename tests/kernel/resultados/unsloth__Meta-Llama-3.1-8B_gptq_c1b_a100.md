> Rodada do notebook `strata_8b_gptq_linha_colab.ipynb` (commit 6b6ed52), interrompida por falta de unidades de computação do Colab depois do P: os casos DL e DC não rodaram. O que saiu equivale ao C1b (Hessianas com 1000 trechos, KL em 200 trechos).

# STRATA, C1: bits dos níveis escolhidos com a Hessiana (Llama-3.1-8B, caminho F16)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. KL contra o BF16 em 200 trechos de 512 tokens de cada texto (wikitext-2 raw de teste; C4 en, validação). imatrix com 500 e Hessianas com 1000 trechos da calibração misturada (wikitext de treino + C4 validação 00001); amortecimento 0.1. Matrizes das camadas: 6.98 bilhões de pesos. Nível 0 = Q4_K_M com imatrix (4.583 GiB); RAM do nível 1 5.395 GiB, do nível 2 6.208 GiB.

## KL (×1000)

| modelo | GiB | wikitext | C4 | mesmo 1º token (wiki) |
|---|---:|---:|---:|---:|
| Q4_K_M_imat | 4.583 | 22.92 ± 0.28 | 37.03 ± 1.07 | 92.92% |
| Q5_K_S_imat | 5.215 | 7.70 ± 0.12 | 10.55 ± 0.31 | 95.82% |
| Q5_K_M_imat | 5.339 | 6.97 ± 0.09 | 10.34 ± 0.30 | 96.09% |
| Q6_K_imat | 6.143 | 3.06 ± 0.03 | 4.31 ± 0.13 | 97.28% |
| Q8_0 | 7.954 | 0.52 ± 0.01 | 0.67 ± 0.02 | 98.97% |
| P_perto_N1 | 5.395 | 9.61 ± 0.12 | 13.43 ± 0.37 | 95.36% |
| P_perto_N2 | 6.208 | 3.34 ± 0.03 | 4.41 ± 0.11 | 97.17% |
| P_gptq_N1 | 5.395 | 6.49 ± 0.10 | 10.41 ± 0.33 | 96.29% |
| P_gptq_N2 | 6.208 | 2.92 ± 0.03 | 4.35 ± 0.13 | 97.34% |

## Régua (bits/peso a mais que a frente dos k-quants em F16, no mesmo tamanho)

| candidato | nível | wikitext | C4 | ganho sobre o P de hoje mantido no C4 |
|---|---:|---:|---:|---:|
| P_perto | 1 | +0.380 | +0.371 |  |
| P_perto | 2 | +0.185 | +0.107 |  |
| P_gptq | 1 | -0.016 | +0.119 | 65% |
| P_gptq | 2 | +0.022 | +0.090 | 11% |

**Veredito pela régua (P):** ZONA CINZA: ver a tabela (algum nível entre +0,05 e +0,12, só os D chegam lá, ou o ganho não se mantém no C4).

## Erro de saída das camadas (Σ ΔW H ΔWᵀ na calibração, relativo ao do nível 0)

| candidato | nível 1 | nível 2 |
|---|---:|---:|
| P_perto | 0.4382 | 0.1051 |
| P_gptq | 0.2409 | 0.0739 |

Tempo total: 230 min.
