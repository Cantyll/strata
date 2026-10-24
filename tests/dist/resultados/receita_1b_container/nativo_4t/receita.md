# Receita: bartowski/Llama-3.2-1B-Instruct-GGUF

Requantizado de `Llama-3.2-1B-Instruct-f16.gguf` com `llama-quantize`, 4 threads.

| Variante | Bytes | SHA-256 igual ao do Hub | Receita | Tempo (s) |
|---|---|---|---|---|
| IQ3_M | 657,289,344 | sim | imatrix | 73.0 |
| IQ4_XS | 743,141,504 | sim | imatrix | 65.4 |
| Q3_K_L | 732,524,672 | sim | imatrix | 22.9 |
| Q3_K_XL | 796,139,648 | sim | imatrix | 20.8 |
| Q4_0 | 773,025,920 | sim | imatrix | 12.1 |
| Q4_0_4_4 | 770,928,768 | sim | imatrix | 5.4 |
| Q4_0_4_8 | 770,928,768 | sim | imatrix | 5.6 |
| Q4_0_8_8 | 770,928,768 | sim | imatrix | 4.8 |
| Q4_K_L | 871,309,440 | sim | imatrix | 29.9 |
| Q4_K_M | 807,694,464 | sim | imatrix | 31.5 |
| Q4_K_S | 775,647,360 | sim | imatrix | 34.9 |
| Q5_K_L | 975,118,464 | sim | imatrix | 31.8 |
| Q5_K_M | 911,503,488 | sim | imatrix | 32.6 |
| Q5_K_S | 892,563,584 | sim | imatrix | 35.1 |
| Q6_K | 1,021,800,576 | sim | imatrix | 13.3 |
| Q6_K_L | 1,085,415,552 | sim | imatrix | 12.0 |
| Q8_0 | 1,321,083,008 | sim | imatrix | 5.9 |

17/17 variantes batem; 14.68 GB de GGUF quantizado no repositório.
