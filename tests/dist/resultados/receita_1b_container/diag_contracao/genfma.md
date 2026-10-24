# Receita: bartowski/Llama-3.2-1B-Instruct-GGUF

Requantizado de `Llama-3.2-1B-Instruct-f16.gguf` com `llama-quantize`, 4 threads.

| Variante | Bytes | SHA-256 igual ao do Hub | Receita | Tempo (s) |
|---|---|---|---|---|
| Q4_K_M | 807,694,464 | sim | imatrix | 32.8 |
| IQ4_XS | 743,141,504 | sim | imatrix | 66.4 |

2/2 variantes batem; 1.55 GB de GGUF quantizado no repositório.
