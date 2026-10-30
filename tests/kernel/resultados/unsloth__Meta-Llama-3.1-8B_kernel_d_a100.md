# STRATA, kernel do tipo D na velocidade do P (Llama-3.1-8B, llama.cpp CUDA)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. Conferência do CUDA: tudo OK. KL contra o BF16 nos 80 primeiros trechos do wikitext-2 raw de teste; imatrix com 125 trechos do texto de treino. Velocidade: tg128, mediana de 3 rodadas intercaladas de 5 repetições. Antes: STRATA `f441834`; depois: `256d05e`; llama.cpp `2a66703`.

## Velocidade de geração (tok/s)

| modelo | depois | depois sem fusão | antes | rodadas (depois) |
|---|---:|---:|---:|---|
| Q5_K_M_imat | 147.6 | – | – | 147.7, 147.4, 147.6 |
| Q6_K_imat | 120.8 | – | – | 120.8, 120.6, 120.8 |
| P_N1 | 142.4 | 138.7 | 136.1 | 142.4, 142.2, 142.4 |
| P_N2 | 147.3 | 137.5 | 142.6 | 147.2, 147.3, 147.3 |
| D_N1 | 125.3 | 121.6 | 92.0 | 125.3, 125.6, 125.2 |
| D_N2 | 120.1 | 113.1 | 93.5 | 120.1, 119.9, 120.1 |
| DL_N1 | 136.1 | 135.9 | – | 136.1, 136.2, 136.0 |
| DL_N2 | 143.5 | 132.6 | – | 143.5, 143.0, 143.5 |

## Régua de velocidade

| candidato | N1 / P | N1 / Q5_K_M | N2 / P | N2 / Q6_K | veredito |
|---|---:|---:|---:|---:|---|
| D exato, depois | 0.88 | 0.85 | 0.82 | 0.99 | **reprova** |
| D linha, depois | 0.96 | 0.92 | 0.97 | 1.19 | **zona cinza** |
| D exato, depois sem fusão | 0.88 | 0.82 | 0.82 | 0.94 | **reprova** |
| D linha, depois sem fusão | 0.98 | 0.92 | 0.96 | 1.10 | **zona cinza** |
| D exato, antes | 0.68 | 0.62 | 0.66 | 0.77 | **reprova** |

(Nas linhas "sem fusão" e "antes", o P é o do mesmo binário e modo; o k-quant é sempre o do "depois" com fusão.)

## Qualidade

| modelo | KL ×1000 (desquantização) | KL ×1000 (MMVQ, -ub 8) | mesmo 1º token (desq.) | GGUF (GiB) |
|---|---:|---:|---:|---:|
| Q4_K_M_imat | 27.599 ± 0.599 | – | 92.461% | 4.583 |
| Q5_K_M_imat | 8.973 ± 0.170 | – | 95.672% | 5.339 |
| Q6_K_imat | 3.627 ± 0.073 | – | 97.088% | 6.143 |
| P_N1 | 11.255 ± 0.243 | 11.524 ± 0.248 | 95.294% | 5.395 |
| D_N1 | 9.700 ± 0.207 | 9.821 ± 0.203 | 95.51% | 5.395 |
| DL_N1 | 9.791 ± 0.227 | 9.980 ± 0.228 | 95.441% | 5.395 |
| P_N2 | 3.815 ± 0.078 | 4.022 ± 0.075 | 97.0% | 6.208 |
| D_N2 | 3.529 ± 0.072 | 3.748 ± 0.084 | 97.201% | 6.208 |
| DL_N2 | 3.572 ± 0.072 | 3.802 ± 0.078 | 97.152% | 6.208 |

| conferência | nível 1 | nível 2 | régua | resultado |
|---|---:|---:|---|---|
| D antes → depois, MMVQ | 9.798 → 9.821 (+0.08 ±) | 3.760 → 3.748 (-0.10 ±) | dentro de 1 ± | **passa** |
| D linha / D, desquantização | 1.009 | 1.012 | ≤ 1,03 | **passa** |
| fração do ganho do D sobre o P que o linha guarda | 0.94 | 0.85 | (1 = todo, 0 = nada) | |
