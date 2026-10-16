Modelo `unsloth/Llama-3.2-1B`, 112 matrizes (973,1 M pesos). Arquivo sem perdas = estrato 0 k-quant (bytes reais do GGUF, llama-quantize sem imatrix) + resíduo até o BF16 exato, condicionado à célula do k-quant.

| Estrato 0 | Bits/peso do estrato 0 | Resíduo | Total sem perdas | vs STRATA sem perdas | Limite com histograma ideal |
| --- | ---: | ---: | ---: | ---: | ---: |
| sinal+expoente (STRATA 0.2) | 3,60 | 6,99 | 10.594 | | |
| Q4_K_M | 4.802 | 6.434 | 11.236 | +0.642 | 11.187 |
| Q5_K_M | 5.656 | 5.571 | 11.227 | +0.633 | 11.183 |
| Q6_K | 6.562 | 4.637 | 11.199 | +0.605 | 11.181 |

Resíduo: sinal+expoente por oitava do passo da grade (ordem 0, tabela no cabeçalho, contada) e mantissa uniforme; soma exata de -log2 p. Nenhum peso caiu fora da célula do k-quant.
