Matrizes das camadas: 2.819 bilhões de pesos. RAM do nível 0 (Q4_K_M) 1918.4 MiB, nível 1 2254.4, nível 2 2590.4.

## Frente e níveis inteiros (KL ×1000, PyTorch, 40 trechos)

| modelo | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 1918.4 | 23.89 ± 0.71 |  | 23.35 ± 1.16 |  |
| Q5_K_S | 2156.9 | 7.27 ± 0.16 |  | 6.78 ± 0.19 |  |
| Q5_K_M | 2207.1 | 8.46 ± 0.55 |  | 8.74 ± 0.73 |  |
| Q6_K | 2513.9 | 3.99 ± 0.30 |  | 3.90 ± 0.29 |  |
| Q8_0 | 3255.9 | 0.35 ± 0.01 |  | 0.32 ± 0.01 |  |
| nível 1 | 2254.4 | 5.55 ± 0.12 | -0.186 | 5.98 ± 0.25 | +0.048 |
| nível 2 | 2590.4 | 2.05 ± 0.04 | -0.375 | 2.32 ± 0.23 | -0.229 |

## Misturas (KL ×1000)

| orçamento (GiB) | alocador | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais | níveis inteiros (wiki / C4) | bits dos inteiros |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2.1104 | medido | 2150.9 | 8.28 ± 0.18 | +0.060 | 8.52 ± 0.48 | +0.113 | 8.70 / 9.10 | +0.090 / +0.151 |
| 2.1104 | de hoje | 2153.5 | 17.20 ± 1.13 | +0.504 | 15.35 ± 0.94 | +0.459 | 8.61 / 9.00 | +0.091 / +0.152 |
| 2.2416 | medido | 2284.4 | 4.70 ± 0.10 | -0.392 | 5.10 ± 0.25 | -0.169 | 5.08 / 5.50 | -0.255 / -0.024 |
| 2.2416 | de hoje | 2287.7 | 12.69 ± 0.98 | +0.722 | 11.29 ± 0.72 | +0.682 | 5.03 / 5.45 | -0.262 / -0.033 |
| 2.3729 | medido | 2419.4 | 3.18 ± 0.06 | -0.487 | 3.56 ± 0.21 | -0.362 | 3.40 / 3.76 | -0.425 / -0.314 |
| 2.3729 | de hoje | 2422.4 | 6.04 ± 0.15 | +0.461 | 6.62 ± 0.39 | +0.744 | 3.37 / 3.73 | -0.424 / -0.312 |

## A soma prevê a medida? (textos da medida)

| orçamento | prevista | medida | erro |
|---|---:|---:|---:|
| 2.1104 | 5.998 | 5.973 | +0.4% |
| 2.2416 | 3.581 | 3.627 | -1.3% |
| 2.3729 | 2.287 | 2.464 | -7.2% |

**Veredito do C2 pela régua:** SUCESSO: ≤ +0,10 bit além dos níveis inteiros nos dois textos, nenhum orçamento pior que o de hoje e a soma prevista erra ≤ 10%.

Ganho do medido sobre os níveis inteiros, em bits (wiki / C4): 2.1104 GiB +0.030 / +0.038, 2.2416 GiB +0.138 / +0.145, 2.3729 GiB +0.062 / +0.048.

**C1: o 3B repete o atraso do 8B?** Nível 1 no C4: +0.048 bit (regra: ≥ +0,08 repete o 8B; ≤ +0,04 promete demais como o 1B).
