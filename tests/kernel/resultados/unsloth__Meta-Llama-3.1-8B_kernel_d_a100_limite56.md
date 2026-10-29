# STRATA, kernel do tipo D na velocidade do P (Llama-3.1-8B, llama.cpp CUDA)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. Conferência do CUDA: tudo OK. KL contra o BF16 nos 80 primeiros trechos do wikitext-2 raw de teste; imatrix com 125 trechos do texto de treino. Velocidade: tg128, mediana de 3 rodadas intercaladas de 5 repetições. Antes: STRATA `f441834`; depois: `9756817`; llama.cpp `2a66703`.

## Velocidade de geração (tok/s)

| modelo | depois | depois sem fusão | antes | rodadas (depois) |
|---|---:|---:|---:|---|
| Q5_K_M_imat | 146.2 | – | – | 146.4, 146.2, 146.2 |
| Q6_K_imat | 119.7 | – | – | 119.7, 119.7, 119.7 |
| P_N1 | 141.3 | 137.7 | 134.8 | 141.2, 141.3, 141.5 |
| P_N2 | 145.9 | 136.6 | 141.5 | 145.9, 145.9, 145.8 |
| D_N1 | 123.4 | 120.9 | 91.5 | 123.8, 123.4, 123.4 |
| D_N2 | 121.6 | 112.4 | 93.0 | 121.8, 121.5, 121.6 |
| DL_N1 | 137.9 | 134.7 | – | 137.8, 137.9, 138.0 |
| DL_N2 | 140.6 | 132.1 | – | 140.4, 140.6, 140.6 |

## Régua de velocidade

| candidato | N1 / P | N1 / Q5_K_M | N2 / P | N2 / Q6_K | veredito |
|---|---:|---:|---:|---:|---|
| D exato, depois | 0.87 | 0.84 | 0.83 | 1.02 | **reprova** |
| D linha, depois | 0.98 | 0.94 | 0.96 | 1.17 | **zona cinza** |
| D exato, depois sem fusão | 0.88 | 0.83 | 0.82 | 0.94 | **reprova** |
| D linha, depois sem fusão | 0.98 | 0.92 | 0.97 | 1.10 | **zona cinza** |
| D exato, antes | 0.68 | 0.63 | 0.66 | 0.78 | **reprova** |

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
