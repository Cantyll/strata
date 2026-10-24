# Teste de armazenamento: unsloth/Meta-Llama-3.1-8B-Instruct

GGUF base: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (sem mudança) · K = 3 · 8 núcleos

| Arquivo | bytes |
|---|---:|
| `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.920.739.232 |
| `model-e00-00000-of-00001.strata` | 265.496 |
| `model-e01-00000-of-00001.strata` | 1.003.749.392 |
| `model-e02-00000-of-00001.strata` | 1.003.749.392 |
| `model-e03-00000-of-00001.strata` | 1.003.749.392 |
| `model-e04-00000-of-00001.strata` | 3.243.137.938 |
| `índice` | 325.155 |

```
Q4_K:  6498 M pesos  estrato 0 4.500 + níveis 3.000 + fecho 3.611 = 11.111 bits/peso  (41089 escapes)
Q6_K:  1532 M pesos  estrato 0 6.562 + níveis 3.000 + fecho 1.619 = 11.182 bits/peso  (141369 escapes)
camadas:  6979 M pesos  estrato 0 4.797 + níveis 3.000 + fecho 3.327 = 11.124 bits/peso  (118150 escapes)
total:  8030 M pesos  estrato 0 4.893 + níveis 3.000 + fecho 3.231 = 11.125 bits/peso  (182458 escapes)
```

- **Total:** 11.18 GB = 11.134 bits/peso (8.0 G pesos em matrizes)
- **Volta:** 4/4 shards com o SHA-256 do Hugging Face
- **Hub hoje (BF16 + Q4_K_M + Q5_K_M + Q6_K):** 33.31 GB → 11.18 GB (66.4% menos)
- **Quem já tem o Q4_K_M e quer o BF16 exato:** baixa 6.25 GB em vez de 16.06 GB (61.1% menos)
- **Repositório de GGUF:** 24 variantes, 143.45 GB; as 3 cobertas são 12%
- **Tempos:** GGUF 16 s, shards 59 s (download), gravar 886 s, voltar 282 s
- **Pico de RAM:** gravar 5.0 GB, voltar 3.1 GB

Q5_K_M 5.73 GB · Q6_K 6.60 GB · Q8_0 8.54 GB
