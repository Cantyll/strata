# STRATA com escala própria nos níveis (Llama-3.1-8B, só qualidade, matrizes em F16)

GPU: NVIDIA L4, 23034 MiB, 8.9, 580.82.07. KL contra o BF16 nos 100 primeiros trechos do wikitext-2 raw de teste; imatrix com 125 trechos do texto de treino. Todas as matrizes das camadas em F16 (k-quants desquantizados), então as ativações não são arredondadas para 8 bits em nenhum modelo.

| modelo | bits extras por nível | KL (×1000) | mesmo 1º token |
|---|---:|---:|---:|
| Q5_K_M_imat |  | 7.48 ± 0.13 | 95.99% |
| Q6_K_imat |  | 3.25 ± 0.06 | 97.24% |
| STRATA_fixo_N1 | 0 | 10.72 ± 0.20 | 95.29% |
| STRATA_fixo_N2 | 0 | 3.65 ± 0.06 | 97.06% |
| STRATA_bloco_N1 | ~0,016 | 8.81 ± 0.17 | 95.59% |
| STRATA_bloco_N2 | ~0,016 | 3.48 ± 0.06 | 97.16% |

Relação com o k-quant com imatrix do mesmo nível (abaixo de 1 = mais perto do original):

| comparação | KL |
|---|---:|
| STRATA_fixo_N1 / Q5_K_M_imat | 1.43 |
| STRATA_fixo_N2 / Q6_K_imat | 1.12 |
| STRATA_bloco_N1 / Q5_K_M_imat | 1.18 |
| STRATA_bloco_N2 / Q6_K_imat | 1.07 |

Llama-3.2-1B no container, mesmo caminho F16 (seção 26): Q5_K_M 11,79, Q6_K 5,37; passo fixo 15,87 / 4,43; bloco 12,41 / 4,20; sub-bloco 11,68 / 4,05.
