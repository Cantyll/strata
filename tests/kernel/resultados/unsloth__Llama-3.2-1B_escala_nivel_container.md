# Níveis com escala própria (Llama-3.2-1B, container, só qualidade)

Matrizes das camadas em F16 (o llama.cpp não arredonda as ativações para 8 bits nesse caminho), KL contra o BF16 nos 40
primeiros trechos de 512 tokens do wikitext-2 raw de teste, imatrix de 125 trechos do texto de treino. Nível 0 = Q4_K_M com
imatrix em todos os STRATA. `tests/kernel/strata_escala_nivel.py`; o modo `fixo` dá pesos iguais bit a bit aos dos tipos P.

| modelo | bits extras por nível | nível 1 KL ×1000 | nível 2 KL ×1000 |
|---|---:|---:|---:|
| Q4_K_M com imatrix (nível 0) | | 38,76 ± 0,63 | |
| Q5_K_M com imatrix | | 11,79 ± 0,22 | |
| Q6_K com imatrix | | | 5,37 ± 0,12 |
| STRATA de hoje (passo fixo, m = 8) | 0 | 15,87 ± 0,24 | 4,43 ± 0,07 |
| passo fixo m = 7 | 0 | 13,94 ± 0,21 | 4,20 ± 0,07 |
| escala por bloco de 256 | 0,016 | 12,41 ± 0,19 | 4,20 ± 0,07 |
| escala por sub-bloco, só Q4_K | 0,107 | 12,47 ± 0,19 | 4,27 ± 0,07 |
| escala por sub-bloco, Q4_K e Q6_K | 0,143 | **11,68 ± 0,18** | **4,05 ± 0,07** |

Mesmos modelos no caminho de verdade (tipos quantizados, ativações em 8 bits): Q4_K_M 40,94, Q5_K_M 13,87, Q6_K 7,28,
STRATA de hoje 17,75 / 6,40. O arredondamento das ativações soma ~2 (×1000) a todos.
