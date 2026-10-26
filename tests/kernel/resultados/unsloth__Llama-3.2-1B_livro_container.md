# Livro de códigos aninhado, Llama-3.2-1B (contêiner, caminho F16, 40 trechos)

Mesmas contas da célula de régua do notebook do 8B. Fronteira da régua = k-quants com imatrix (incl. Q4_K_S, Q5_K_S, Q6_K+Q8_0) + IQ4_XS.

| modelo | bits/peso | KL ×1000 | mesmo 1º token | erro ponderado |
|---|---:|---:|---:|---:|
| F16_Q3_K_M_imat | 3.842 | 127.89 ± 2.03 | 82.52% | 1.8509% |
| linha_N0 | 4.099 | 174.58 ± 2.67 | 79.37% | 1.3026% |
| bloco_N0 | 4.250 | 49.46 ± 1.08 | 89.05% | 0.4718% |
| F16_IQ4_XS_imat | 4.272 | 47.68 ± 0.73 | 88.99% | 0.5070% |
| F16_Q4_K_S_imat | 4.539 | 43.05 ± 0.70 | 89.65% | 0.4275% |
| F16_Q4_K_M_imat | 4.802 | 38.76 ± 0.63 | 89.99% | 0.4201% |
| linha_N1 | 5.198 | 43.33 ± 0.73 | 89.54% | 0.3239% |
| bloco_N1 | 5.250 | 17.36 ± 0.34 | 93.04% | 0.1573% |
| bloco_ind_N1 | 5.250 | 12.58 ± 0.25 | 93.84% | 0.1192% |
| F16_Q5_K_S_imat | 5.500 | 11.53 ± 0.21 | 94.54% | 0.1104% |
| F16_Q5_K_M_imat | 5.656 | 11.79 ± 0.22 | 94.41% | 0.1089% |
| bloco_N2 | 6.250 | 4.98 ± 0.08 | 95.95% | 0.0420% |
| bloco_ind_N2 | 6.250 | 4.04 ± 0.06 | 96.52% | 0.0342% |
| linha_N2 | 6.397 | 10.89 ± 0.25 | 94.71% | 0.0740% |
| F16_Q6_K_imat | 6.562 | 5.37 ± 0.12 | 95.93% | 0.0282% |
| F16_Q6_K_Q8mix_imat | 7.431 | 2.17 ± 0.03 | 97.22% | 0.0175% |
| F16_Q8_0 | 8.500 | 0.62 ± 0.02 | 98.43% | 0.0031% |

| candidato | bits/peso | KL ×1000 | bits a mais (curva antiga + IQ4_XS) | bits a mais (k-quants) | bits a mais (fronteira da régua) |
|---|---:|---:|---:|---:|---:|
| bloco_N0 | 4.250 | 49.46 | -0.006 | -0.200 | -0.006 |
| bloco_N1 | 5.250 | 17.36 | -0.128 | -0.014 | -0.014 |
| bloco_N2 | 6.250 | 4.98 | -0.380 | -0.384 | -0.384 |
| bloco_ind_N1 | 5.250 | 12.58 | -0.359 | -0.200 | -0.200 |
| bloco_ind_N2 | 6.250 | 4.04 | -0.568 | -0.585 | -0.585 |
| linha_N0 | 4.099 | 174.58 | +0.393 | +0.457 | +0.393 |
| linha_N1 | 5.198 | 43.33 | +0.682 | +0.664 | +0.676 |
| linha_N2 | 6.397 | 10.89 | +0.649 | +0.817 | +0.817 |

Dominância:
- bloco_N0: nenhum
- bloco_N1: nenhum
- bloco_N2: Q6_K_imat
- linha_N0: nenhum
- linha_N1: nenhum
- linha_N2: nenhum
- bloco_ind_N1: nenhum
- bloco_ind_N2: Q6_K_imat

fronteira: [(3.842, 127.89), (4.272, 47.68), (4.539, 43.05), (4.802, 38.76), (5.5, 11.53), (6.562, 5.37), (7.431, 2.17), (8.5, 0.62)]

## Kernel (CPU do contêiner)

| model                          |       size |     params | backend    | threads | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       1 |   0 |           pp512 |         76.63 ± 3.31 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       1 |   0 |            tg64 |          6.74 ± 0.29 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |   0 |           pp512 |       326.96 ± 23.38 |
| llama 1B Q4_K - Medium         | 762.81 MiB |     1.24 B | CPU        |       4 |   0 |            tg64 |         25.54 ± 0.26 |
| llama 1B IQ4_XS - 4.25 bpw     | 701.25 MiB |     1.24 B | CPU        |       1 |   0 |           pp512 |        108.86 ± 5.05 |
| llama 1B IQ4_XS - 4.25 bpw     | 701.25 MiB |     1.24 B | CPU        |       1 |   0 |            tg64 |          6.95 ± 0.22 |
| llama 1B IQ4_XS - 4.25 bpw     | 701.25 MiB |     1.24 B | CPU        |       4 |   0 |           pp512 |       370.06 ± 34.68 |
| llama 1B IQ4_XS - 4.25 bpw     | 701.25 MiB |     1.24 B | CPU        |       4 |   0 |            tg64 |         24.51 ± 0.72 |
| llama 1B Q5_K - Medium         | 861.81 MiB |     1.24 B | CPU        |       1 |   0 |           pp512 |         97.18 ± 4.57 |
| llama 1B Q5_K - Medium         | 861.81 MiB |     1.24 B | CPU        |       1 |   0 |            tg64 |          6.01 ± 0.12 |
| llama 1B Q5_K - Medium         | 861.81 MiB |     1.24 B | CPU        |       4 |   0 |           pp512 |       299.92 ± 26.48 |
| llama 1B Q5_K - Medium         | 861.81 MiB |     1.24 B | CPU        |       4 |   0 |            tg64 |         21.13 ± 0.60 |
| llama 1B Q6_K                  | 967.00 MiB |     1.24 B | CPU        |       1 |   0 |           pp512 |         83.15 ± 2.02 |
| llama 1B Q6_K                  | 967.00 MiB |     1.24 B | CPU        |       1 |   0 |            tg64 |          6.03 ± 0.15 |
| llama 1B Q6_K                  | 967.00 MiB |     1.24 B | CPU        |       4 |   0 |           pp512 |        297.19 ± 8.54 |
| llama 1B Q6_K                  | 967.00 MiB |     1.24 B | CPU        |       4 |   0 |            tg64 |         21.43 ± 0.15 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | CPU        |       1 |   0 |           pp512 |         36.81 ± 1.42 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | CPU        |       1 |   0 |            tg64 |          4.56 ± 0.08 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | CPU        |       4 |   0 |           pp512 |        104.60 ± 7.62 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | CPU        |       4 |   0 |            tg64 |         17.32 ± 0.42 |

build: 2a66703 (1)

conferência contra a referência escalar: OK
16384 × 4096 pesos; ns por peso (melhor de 7)

| tipo | bits/peso | 1 thread | 4 threads |
|---|---:|---:|---:|
| Q4_K (ggml) | 4.50 | 0.0386 | 0.0091 |
| IQ4_XS (ggml) | 4.25 | 0.0529 | 0.0223 |
| Q5_K (ggml) | 5.50 | 0.0511 | 0.0124 |
| Q6_K (ggml) | 6.56 | 0.0412 | 0.0111 |
| Q8_0 (ggml) | 8.50 | 0.0528 | 0.0200 |
| livro nível 0 (tabela de 16) | 4.25 | 0.0733 | 0.0186 |
| livro nível 1 (tabela de 32) | 5.25 | 0.0934 | 0.0255 |
| livro nível 2 (tabela de 64) | 6.25 | 0.1216 | 0.0286 |
