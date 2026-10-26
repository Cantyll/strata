# Teste de armazenamento: unsloth/Meta-Llama-3.1-70B-Instruct

GGUF base: `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` (sem mudança) · K = 3 · 8 núcleos

| Arquivo | bytes |
|---|---:|
| `Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf` | 42.520.398.400 |
| `model-e00-00000-of-00001.strata` | 530.101.411 |
| `model-e01-00000-of-00001.strata` | 8.777.105.424 |
| `model-e02-00000-of-00001.strata` | 8.777.105.424 |
| `model-e03-00000-of-00001.strata` | 8.777.105.424 |
| `model-e04-00000-of-00001.strata` | 28.891.018.088 |
| `índice` | 793.781 |

```
Q4_K: 59435 M pesos  estrato 0 4.500 + níveis 3.000 + fecho 3.597 = 11.097 bits/peso  (373859 escapes)
Q6_K: 10781 M pesos  estrato 0 6.562 + níveis 3.000 + fecho 1.609 = 11.172 bits/peso  (974929 escapes)
camadas: 68115 M pesos  estrato 0 4.795 + níveis 3.000 + fecho 3.313 = 11.108 bits/peso  (1186179 escapes)
total: 70217 M pesos  estrato 0 4.817 + níveis 3.000 + fecho 3.292 = 11.108 bits/peso  (1348788 escapes)
```

- **Total:** 98.27 GB = 11.197 bits/peso (70.2 G pesos em matrizes)
- **Volta:** 30/30 shards com o SHA-256 do Hugging Face
- **Hub hoje (BF16 + Q4_K_M + Q5_K_M + Q6_K):** 291.47 GB → 98.27 GB (66.3% menos)
- **Quem já tem o Q4_K_M e quer o BF16 exato:** baixa 55.75 GB em vez de 141.11 GB (60.5% menos)
- **Repositório de GGUF:** 23 variantes, 863.29 GB; as 3 cobertas são 17%
- **Tempos:** GGUF 117 s, shards 1694 s (download), gravar 10203 s, voltar 6128 s
- **Pico de RAM:** gravar 10.1 GB, voltar 5.6 GB

Q5_K_M 49.95 GB · Q6_K 57.89 GB · Q8_0 74.98 GB
