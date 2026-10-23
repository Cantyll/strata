# STRATA no llama.cpp (Llama-3.2-1B)

CPU: Intel(R) Xeon(R) CPU @ 2.20GHz, 12 threads lógicas, 6 núcleos físicos, RAM 175 GB, AVX-512: True. Só velocidade (qualidade não medida nesta rodada). tok/s = mediana de 3 rodadas alternadas. Rodada de 2026-10-19 no Colab Pro (CPU com RAM alta), já com os tipos do STRATA no caminho tiled da leitura de prompt.

| modelo | GGUF (GiB) | PPL | ΔPPL | KL (×1000) | mesmo 1º token | tg128 t1 (tok/s) | tg128 t6 (tok/s) | tg128 t12 (tok/s) | pp512 t6 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 2.309 | nan ± nan |  |  |  |  |  |  |  |
| Q4_K_M | 0.752 | nan ± nan |  |  |  | 8.3 | 39.4 | 47.6 | 255.1 |
| Q5_K_M | 0.849 | nan ± nan |  |  |  | 6.7 | 32.1 | 40.2 | 346.1 |
| STRATA_N1 | 0.882 | nan ± nan |  |  |  | 6.5 | 31.5 | 39.2 | 325.1 |
| Q6_K | 0.952 | nan ± nan |  |  |  | 7.0 | 33.1 | 40.4 | 320.5 |
| STRATA_N2 | 0.979 | nan ± nan |  |  |  | 6.0 | 29.1 | 36.2 | 324.1 |

Velocidade do STRATA em relação ao k-quant de mesma qualidade (1,00 = igual; abaixo de 1 = mais lento):

| comparação | tg128 t1 | tg128 t6 | tg128 t12 | pp512 t6 |
|---|---:|---:|---:|---:|
| STRATA_N1 / Q5_K_M | 0.97 | 0.98 | 0.98 | 0.94 |
| STRATA_N2 / Q6_K | 0.86 | 0.88 | 0.90 | 1.01 |

Qualidade: ver a rodada anterior (`unsloth__Llama-3.2-1B_llamacpp_epyc.md`) e a seção 23 dos resultados.
