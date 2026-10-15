Modelo `unsloth/Llama-3.2-1B`, 288627 tokens do wikitext-2 (teste), janelas de 2048, contas em float32. Cortadas 112 matrizes = 78.7% dos parâmetros (sem embedding nem normas).

| Configuração | Bits/peso no disco | Bits/peso na RAM | Erro rel. nos pesos | PPL | ΔPPL | ΔPPL % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| INT8 progressivo, 4 bits | 4.137 | 4.5 | 9.151% | 13.2887 | +3.5378 | +36.28% |
| INT8 progressivo, 5 bits | 5.137 | 5.5 | 4.609% | 10.2683 | +0.5174 | +5.31% |
| INT8 progressivo, 6 bits | 6.137 | 6.5 | 2.420% | 9.8622 | +0.1113 | +1.14% |
| INT8 progressivo, 7 bits | 7.137 | 7.5 | 1.456% | 9.8007 | +0.0498 | +0.51% |
| INT8 progressivo, 8 bits | 8.137 | 8.5 | 0.549% | 9.7544 | +0.0035 | +0.04% |
| BF16 original (estrato 7) | 11.114 | 16.0 | 0.000% | 9.7509 | +0.0000 | +0.00% |

STRATA: disco medido com deflate-raw por plano (como o core); na RAM o corte volta a BF16 (16 bits), sem kernel próprio.
INT8 progressivo: um só arquivo INT8 por bloco de 32 em sinal+magnitude; cortar planos dá k bits. Disco medido (escala + sinal + planos, deflate-raw); RAM nominal k + 0,5.
INT-k: disco e RAM nominais (k bits + escala FP16 a cada 32 pesos), sem medir compressão extra.
