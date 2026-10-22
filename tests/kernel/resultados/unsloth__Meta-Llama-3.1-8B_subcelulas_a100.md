# STRATA, sub-células por tensor no kernel (Llama-3.1-8B, llama.cpp CUDA)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. Conferência do CUDA: tudo OK. KL contra o BF16 nos 100 primeiros trechos do wikitext-2 raw de teste; imatrix com 125 trechos do texto de treino. Matrizes das camadas: 6.98 bilhões de pesos, 1.01 bilhões em Q6_K.

## KL e tamanho

| modelo | GGUF (GiB) | KL (×1000) | mesmo 1º token |
|---|---:|---:|---:|
| Q3_K_M_imat | 3.743 | 77.94 ± 1.21 | 87.49% |
| Q4_K_M_imat | 4.583 | 26.26 ± 0.49 | 92.56% |
| Q5_K_M_imat | 5.339 | 8.60 ± 0.14 | 95.77% |
| Q6_K_imat | 6.143 | 3.47 ± 0.06 | 97.16% |
| Q8_0 | 7.954 | 0.79 ± 0.01 | 98.69% |
| P_N1 | 5.512 | 10.78 ± 0.20 | 95.33% |
| P_N2 | 6.208 | 3.66 ± 0.06 | 97.02% |
| D_N1 | 5.512 | 9.27 ± 0.17 | 95.59% |
| D_N2 | 6.208 | 3.40 ± 0.06 | 97.23% |
| F16_Q4_K_M_imat | 4.583 | 25.32 ± 0.49 | 92.68% |
| F16_Q5_K_M_imat | 5.339 | 7.55 ± 0.14 | 96.01% |
| F16_Q6_K_imat | 6.143 | 3.31 ± 0.06 | 97.12% |
| F16_Q8_0 | 7.954 | 0.57 ± 0.01 | 98.96% |
| F16_fixo_N1 | 13.685 | 10.78 ± 0.20 | 95.33% |
| F16_fixo_N2 | 13.685 | 3.66 ± 0.06 | 97.02% |
| F16_codigo_N1 | 13.685 | 9.03 ± 0.17 | 95.47% |
| F16_codigo_N2 | 13.685 | 3.35 ± 0.06 | 97.23% |
| F16_borda_N1 | 13.685 | 9.24 ± 0.17 | 95.56% |
| F16_borda_N2 | 13.685 | 3.39 ± 0.06 | 97.27% |

## Régua (curva = k-quants com imatrix em F16)

| candidato | nível | RAM (GiB) | KL ×1000 | k-quant F16 no mesmo tamanho | razão | bits/peso a mais | com Q6_K empacotado |
|---|---:|---:|---:|---:|---:|---:|---:|
| P_N1 | 1 | 5.512 | 10.78 ± 0.20 | 6.32 | 1.71 | +0.487 | +0.343 |
| D_N1 | 1 | 5.512 | 9.27 ± 0.17 | 6.32 | 1.47 | +0.371 | +0.227 |
| F16_fixo_N1 | 1 | 5.512 | 10.78 ± 0.20 | 6.32 | 1.71 | +0.487 | +0.343 |
| F16_codigo_N1 | 1 | 5.512 | 9.03 ± 0.17 | 6.32 | 1.43 | +0.351 | +0.206 |
| F16_borda_N1 | 1 | 5.512 | 9.24 ± 0.17 | 6.32 | 1.46 | +0.368 | +0.224 |
| P_N2 | 2 | 6.208 | 3.66 ± 0.06 | 3.10 | 1.18 | +0.202 | +0.202 |
| D_N2 | 2 | 6.208 | 3.40 ± 0.06 | 3.10 | 1.10 | +0.114 | +0.114 |
| F16_fixo_N2 | 2 | 6.208 | 3.66 ± 0.06 | 3.10 | 1.18 | +0.202 | +0.202 |
| F16_codigo_N2 | 2 | 6.208 | 3.35 ± 0.06 | 3.10 | 1.08 | +0.097 | +0.097 |
| F16_borda_N2 | 2 | 6.208 | 3.39 ± 0.06 | 3.10 | 1.09 | +0.111 | +0.111 |

Seção 28 (mesmo 8B, emulação): hoje +0,34 / +0,20 bits/peso a mais (níveis 1 / 2, nível 1 com Q6_K empacotado); sub-células por tensor +0,21 / +0,10.

## Kernel contra a emulação (mesma conta; só o arredondamento dos pesos para F16 muda)

| nível | D (kernel) | emulação por borda | diferença em ± |
|---:|---:|---:|---:|
| 1 | 9.27 | 9.24 | +0.1 |
| 2 | 3.40 | 3.39 | +0.1 |

## Velocidade na GPU (tok/s)

| modelo | geração (tg128) | prompt (pp512) |
|---|---:|---:|
| Q5_K_M_imat | 146.0 | 4505 |
| Q6_K_imat | 118.6 | 4264 |
| P_N1 | 145.4 | 6723 |
| P_N2 | 142.2 | 6866 |
| D_N1 | 94.1 | 6613 |
| D_N2 | 93.1 | 6423 |
| D_N1 / P_N1 | 0.65 | 0.98 |
| D_N2 / P_N2 | 0.65 | 0.94 |
| D_N1 / Q5_K_M_imat | 0.64 | 1.47 |
| D_N2 / Q6_K_imat | 0.79 | 1.51 |
