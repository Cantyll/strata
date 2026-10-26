# Receita: bartowski/Llama-3.2-1B-Instruct-GGUF

Requantizado de `Llama-3.2-1B-Instruct-f16.gguf` com `llama-quantize`, 4 threads.

| Variante | Bytes | SHA-256 igual ao do Hub | Receita | Tempo (s) |
|---|---|---|---|---|
| IQ3_M | 657,289,344 | sim | imatrix | 75.6 |
| IQ4_XS | 743,141,504 | sim | imatrix | 65.6 |
| Q4_0 | 773,025,920 | sim | imatrix | 12.0 |
| Q4_K_M | 807,694,464 | sim | imatrix | 30.8 |
| Q6_K | 1,021,800,576 | sim | imatrix | 13.5 |

5/5 variantes batem; 4.00 GB de GGUF quantizado no repositório.
