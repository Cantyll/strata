# Receita: bartowski/Llama-3.2-1B-Instruct-GGUF

Requantizado de `Llama-3.2-1B-Instruct-f16.gguf` com `llama-quantize`, 1 threads.

| Variante | Bytes | SHA-256 igual ao do Hub | Receita | Tempo (s) |
|---|---|---|---|---|
| IQ3_M | 657,289,344 | sim | imatrix | 270.0 |
| IQ4_XS | 743,141,504 | sim | imatrix | 248.7 |
| Q3_K_XL | 796,139,648 | sim | imatrix | 70.1 |
| Q4_K_M | 807,694,464 | sim | imatrix | 119.6 |
| Q6_K | 1,021,800,576 | sim | imatrix | 43.4 |

5/5 variantes batem; 4.03 GB de GGUF quantizado no repositório.
