Modelo `unsloth/Llama-3.2-1B`, 288627 tokens de teste, janelas de 2048, contas em float32. Estrato 0 = Q4_K_M do llama-quantize; nível k = célula do k-quant dividida em 2^k partes, valor no centro.

| Configuração | Bits/peso (nível: nominal) | PPL | ΔPPL % |
| --- | ---: | ---: | ---: |
| Q4_K_M (estrato 0) | 4.802 | 10.1473 | +4.07% |
| llama.cpp Q5_K_M | 5.656 | 9.8731 | +1.25% |
| Q4_K_M + 1 nível(is) | 5.802 | 9.8647 | +1.17% |
| llama.cpp Q6_K | 6.562 | 9.7986 | +0.49% |
| Q4_K_M + 2 nível(is) | 6.802 | 9.7809 | +0.31% |
| Q4_K_M + 3 nível(is) | 7.802 | 9.7611 | +0.11% |
| BF16 original | 16.000 | 9.7509 | +0.00% |
