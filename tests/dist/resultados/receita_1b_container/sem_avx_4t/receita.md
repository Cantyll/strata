# Receita: bartowski/Llama-3.2-1B-Instruct-GGUF

Requantizado de `Llama-3.2-1B-Instruct-f16.gguf` com `llama-quantize`, 4 threads.

| Variante | Bytes | SHA-256 igual ao do Hub | Receita | Tempo (s) |
|---|---|---|---|---|
| IQ3_M | 657,289,344 | **não** | - | - |
| IQ4_XS | 743,141,504 | **não** | - | - |
| Q3_K_L | 732,524,672 | **não** | - | - |
| Q3_K_XL | 796,139,648 | **não** | - | - |
| Q4_0 | 773,025,920 | **não** | - | - |
| Q4_0_4_4 | 770,928,768 | **não** | - | - |
| Q4_0_4_8 | 770,928,768 | **não** | - | - |
| Q4_0_8_8 | 770,928,768 | **não** | - | - |
| Q4_K_L | 871,309,440 | **não** | - | - |
| Q4_K_M | 807,694,464 | **não** | - | - |
| Q4_K_S | 775,647,360 | **não** | - | - |
| Q5_K_L | 975,118,464 | **não** | - | - |
| Q5_K_M | 911,503,488 | **não** | - | - |
| Q5_K_S | 892,563,584 | **não** | - | - |
| Q6_K | 1,021,800,576 | **não** | - | - |
| Q6_K_L | 1,085,415,552 | **não** | - | - |
| Q8_0 | 1,321,083,008 | sim | imatrix | 8.2 |

1/17 variantes batem; 14.68 GB de GGUF quantizado no repositório.
