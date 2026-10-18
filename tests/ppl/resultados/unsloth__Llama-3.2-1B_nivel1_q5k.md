Modelo `unsloth/Llama-3.2-1B`, 288627 tokens do wikitext-2 (teste), janelas de 2048, contas em float32. Cortadas 112 matrizes = 78.7% dos parâmetros (sem embedding nem normas).

| Configuração | Bits/peso no disco | Bits/peso na RAM | Erro rel. nos pesos | PPL | ΔPPL | ΔPPL % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp Q4_K_M | 4.802 | 4.8 | 6.457% | 10.1473 | +0.3965 | +4.07% |
| llama.cpp Q5_K_M | 5.656 | 5.7 | 3.401% | 9.8731 | +0.1222 | +1.25% |
| llama.cpp STRATA_nivel1_Q5K | 5.656 | 5.7 | 3.862% | 9.8908 | +0.1399 | +1.43% |
| BF16 original (estrato 7) | 10.594 | 16.0 | 0.000% | 9.7509 | +0.0000 | +0.00% |

STRATA: disco medido por plano como o core v0.2 (rANS no expoente, deflate-raw nos outros); na RAM o corte volta a BF16 (16 bits), sem kernel próprio.
INT8 progressivo: um só arquivo INT8 por bloco de 32 em sinal+magnitude; cortar planos dá k bits. Disco medido (escala + sinal + planos, deflate-raw); RAM nominal k + 0,5.
INT-k: disco e RAM nominais (k bits + escala FP16 a cada 32 pesos), sem medir compressão extra.
llama.cpp: bytes reais das mesmas matrizes no GGUF (llama-quantize, sem imatrix), desquantizados pelo gguf-py; embedding e normas em BF16 como nas outras linhas. Tipos por config: Q4_K_M Q4_K 85%, Q6_K 15%; Q5_K_M Q5_K 85%, Q6_K 15%; STRATA_nivel1_Q5K Q5_K 85%, Q6_K 15%.
