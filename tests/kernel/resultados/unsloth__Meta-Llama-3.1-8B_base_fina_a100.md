# STRATA com base fina (Llama-3.1-8B, llama.cpp CUDA)

GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0, 580.82.07. wikitext-2 raw de teste, contexto 512, inteiro; KL contra o BF16 nos 100 primeiros trechos; imatrix com 125 trechos do texto de treino. FINA2 = base fina T=2, pesos 1, 2, 4; FINA1 = base fina T=1.

| modelo | GGUF (GiB) | PPL | ΔPPL | KL (×1000) | mesmo 1º token |
|---|---:|---:|---:|---:|---:|
| BF16 | 14.965 | 6.4010 ± 0.0394 | +0.00% |  |  |
| Q3_K_M_imat | 3.743 | 6.8911 ± 0.0432 | +7.66% | 77.94 ± 1.21 | 87.49% |
| Q4_K_M_imat | 4.583 | 6.5654 ± 0.0407 | +2.57% | 26.26 ± 0.49 | 92.56% |
| FINA2_N0 | 4.583 | 6.6084 ± 0.0412 | +3.24% | 34.24 ± 0.60 | 91.65% |
| FINA1_N0 | 4.583 | 6.6206 ± 0.0412 | +3.43% | 33.11 ± 0.58 | 91.90% |
| Q5_K_M_imat | 5.339 | 6.4531 ± 0.0398 | +0.81% | 8.60 ± 0.14 | 95.77% |
| FINA2_N1 | 5.512 | 6.4523 ± 0.0398 | +0.80% | 8.69 ± 0.16 | 95.81% |
| FINA1_N1 | 5.512 | 6.4508 ± 0.0398 | +0.78% | 7.59 ± 0.12 | 96.03% |
| Q6_K_imat | 6.143 | 6.4210 ± 0.0396 | +0.31% | 3.47 ± 0.06 | 97.16% |
| FINA2_N2 | 6.208 | 6.4177 ± 0.0395 | +0.26% | 3.16 ± 0.05 | 97.18% |

STRATA contra o k-quant com imatrix do mesmo nível (abaixo de 1 = mais perto do original):

| comparação | tamanho | KL |
|---|---:|---:|
| FINA2_N0 / Q4_K_M_imat | 1.000 | 1.30 |
| FINA1_N0 / Q4_K_M_imat | 1.000 | 1.26 |
| FINA2_N1 / Q5_K_M_imat | 1.032 | 1.01 |
| FINA1_N1 / Q5_K_M_imat | 1.032 | 0.88 |
| FINA2_N2 / Q6_K_imat | 1.011 | 0.91 |

Rodada anterior (STRATA sobre o Q4_K_M com imatrix, seção 24): nível 1 KL 10,78 vs Q5_K_M 8,60; nível 2 3,66 vs Q6_K 3,47 (×1000).
