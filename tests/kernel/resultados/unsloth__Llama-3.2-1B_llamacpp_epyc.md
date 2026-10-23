# STRATA no llama.cpp (Llama-3.2-1B)

CPU: AMD EPYC 9B14, 44 threads lógicas, 22 núcleos físicos, RAM 181 GB, AVX-512: True. Wikitext-2 de teste (a cópia não-raw do repositório do PyTorch, com `<unk>`: o download do raw falhou nesta rodada; serve para comparar os modelos entre si, não com perplexidades publicadas), contexto 512, inteiro; KL contra o BF16 nos 100 primeiros trechos. tok/s = mediana de 3 rodadas alternadas. Rodada de 2026-10-18 no Colab Pro, antes do caminho rápido de leitura de prompt (`pp512` dos tipos do STRATA ainda pelo `vec_dot`).

| modelo | GGUF (GiB) | PPL | ΔPPL | KL (×1000) | mesmo 1º token | tg128 t1 (tok/s) | tg128 t22 (tok/s) | tg128 t44 (tok/s) | pp512 t22 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 2.309 | 9.5863 ± 0.0639 | +0.00% |  |  |  |  |  |  |
| Q4_K_M | 0.752 | 9.8819 ± 0.0661 | +3.08% | 40.73 ± 0.39 | 89.12% | 27.7 | 117.1 | 87.1 | 1245.2 |
| Q5_K_M | 0.849 | 9.7154 ± 0.0650 | +1.35% | 13.75 ± 0.12 | 93.69% | 19.0 | 102.8 | 76.2 | 1696.1 |
| STRATA_N1 | 0.882 | 9.6502 ± 0.0644 | +0.67% | 14.05 ± 0.12 | 93.37% | 19.1 | 99.8 | 74.6 | 512.5 |
| Q6_K | 0.952 | 9.6351 ± 0.0643 | +0.51% | 7.02 ± 0.06 | 95.24% | 24.7 | 94.1 | 68.9 | 1526.2 |
| STRATA_N2 | 0.979 | 9.6173 ± 0.0641 | +0.32% | 5.11 ± 0.04 | 95.94% | 18.7 | 91.2 | 67.1 | 505.0 |

Velocidade do STRATA em relação ao k-quant de mesma qualidade (1,00 = igual; abaixo de 1 = mais lento):

| comparação | tg128 t1 | tg128 t22 | tg128 t44 | pp512 t22 |
|---|---:|---:|---:|---:|
| STRATA_N1 / Q5_K_M | 1.00 | 0.97 | 0.98 | 0.30 |
| STRATA_N2 / Q6_K | 0.76 | 0.97 | 0.97 | 0.33 |

Emulação anterior (Colab, seção 22 dos resultados): nível 1 +1,18%, nível 2 +0,31%, Q5_K_M +1,25%, Q6_K +0,49%.
