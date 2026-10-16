Modelo `unsloth/Llama-3.2-1B`, 288627 tokens do wikitext-2 (teste), janelas de 2048, contas em float32. Cortadas 112 matrizes = 78.7% dos parâmetros (sem embedding nem normas).

| Configuração | Bits/peso no disco | Bits/peso na RAM | Erro rel. nos pesos | PPL | ΔPPL | ΔPPL % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp Q3_K_M | 3.842 | 3.8 | 12.341% | 11.7375 | +1.9869 | +20.38% |
| llama.cpp Q8_0 | 8.500 | 8.5 | 0.550% | 9.7538 | +0.0032 | +0.03% |
| BF16 original (estrato 7) | 10.594 | 16.0 | 0.000% | 9.7506 | +0.0000 | +0.00% |

STRATA: disco medido por plano como o core v0.2 (rANS no expoente, deflate-raw nos outros); na RAM o corte volta a BF16 (16 bits), sem kernel próprio.
INT8 progressivo: um só arquivo INT8 por bloco de 32 em sinal+magnitude; cortar planos dá k bits. Disco medido (escala + sinal + planos, deflate-raw); RAM nominal k + 0,5.
INT-k: disco e RAM nominais (k bits + escala FP16 a cada 32 pesos), sem medir compressão extra.
llama.cpp: bytes reais das mesmas matrizes no GGUF (llama-quantize, sem imatrix), desquantizados pelo gguf-py; embedding e normas em BF16 como nas outras linhas. Tipos por config: Q3_K_M Q3_K 64%, Q4_K 34%, Q5_K 2%; Q8_0 Q8_0 100%.
