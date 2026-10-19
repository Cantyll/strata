# STRATA na GPU (Llama-3.1-8B, llama.cpp CUDA)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. wikitext-2 raw de teste, contexto 512, inteiro; KL contra o BF16 nos 100 primeiros trechos; imatrix com 125 trechos do texto de treino. tok/s = mediana de 3 rodadas alternadas.

| modelo | GGUF (GiB) | PPL | ΔPPL | KL (×1000) | mesmo 1º token | tg128 (tok/s) | pp512 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 14.965 | 6.4010 ± 0.0394 | +0.00% |  |  |  |  |
| Q4_K_M | 4.583 | 6.5782 ± 0.0406 | +2.77% | 29.47 ± 0.49 | 92.27% | 153.9 | 4713.5 |
| Q5_K_M | 5.339 | 6.4653 ± 0.0399 | +1.00% | 10.18 ± 0.17 | 95.40% | 144.6 | 4569.9 |
| STRATA_N1 | 5.512 | 6.4562 ± 0.0398 | +0.86% | 9.98 ± 0.23 | 95.37% | 144.0 | 6808.2 |
| Q6_K | 6.143 | 6.4227 ± 0.0396 | +0.34% | 4.28 ± 0.07 | 96.78% | 117.4 | 4255.3 |
| STRATA_N2 | 6.208 | 6.4157 ± 0.0396 | +0.23% | 3.56 ± 0.06 | 97.06% | 141.0 | 6791.8 |
| Q4_K_M_imat | 4.583 | 6.5654 ± 0.0407 | +2.57% | 26.26 ± 0.49 | 92.56% |  |  |
| Q5_K_M_imat | 5.339 | 6.4531 ± 0.0398 | +0.81% | 8.60 ± 0.14 | 95.77% |  |  |
| STRATA_N1_imat | 5.512 | 6.4679 ± 0.0399 | +1.05% | 10.78 ± 0.20 | 95.33% |  |  |
| Q6_K_imat | 6.143 | 6.4210 ± 0.0396 | +0.31% | 3.47 ± 0.06 | 97.16% |  |  |
| STRATA_N2_imat | 6.208 | 6.4184 ± 0.0395 | +0.27% | 3.66 ± 0.06 | 97.02% |  |  |

STRATA contra o k-quant de mesma qualidade (velocidade: 1,00 = igual, abaixo de 1 = mais lento; KL: abaixo de 1 = mais perto do original):

| comparação | tamanho | KL | tg128 | pp512 |
|---|---:|---:|---:|---:|
| STRATA_N1 / Q5_K_M | 1.032 | 0.98 | 1.00 | 1.49 |
| STRATA_N2 / Q6_K | 1.011 | 0.83 | 1.20 | 1.60 |
| STRATA_N1_imat / Q5_K_M_imat | 1.032 | 1.25 |  |  |
| STRATA_N2_imat / Q6_K_imat | 1.011 | 1.05 |  |  |

Llama-3.2-1B na CPU (seção 23 dos resultados): nível 1 KL 14,1 vs Q5_K_M 13,8; nível 2 KL 5,1 vs Q6_K 7,0 (×1000).
