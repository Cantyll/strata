Modelo `unsloth/Llama-3.2-1B`, 288627 tokens de teste, janelas de 2048, contas em float32. Estrato 0 = Q4_K_M do llama-quantize; nível k = célula do k-quant dividida em 2^k partes, valor no centro.

| Configuração | Bits/peso (nível: nominal) | PPL | ΔPPL % |
| --- | ---: | ---: | ---: |
| Q4_K_M + 1 nível(is), só matrizes Q4_K (Q4_K_R1) | 5.656 | 9.8798 | +1.32% |
| llama.cpp Q5_K_M | 5.656 | 9.8731 | +1.25% |
| Q4_K_M + 1 nível(is), saturados cortados | 5.802 | 9.8655 | +1.18% |
| Q4_K_M + 2 nível(is), só matrizes Q4_K (Q4_K_R2) | 6.509 | 9.8032 | +0.54% |
| llama.cpp Q6_K | 6.562 | 9.7986 | +0.49% |
| Q4_K_M + 2 nível(is), saturados cortados | 6.802 | 9.7814 | +0.31% |
| BF16 original | 16.000 | 9.7509 | +0.00% |
