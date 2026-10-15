Modelo `unsloth/Llama-3.2-1B`, 288627 tokens do wikitext-2 (teste), janelas de 2048, contas em float32. Cortadas 112 matrizes = 78.7% dos parâmetros (sem embedding nem normas).

| Configuração | Bits/peso no disco | Bits/peso na RAM | Erro rel. nos pesos | PPL | ΔPPL | ΔPPL % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STRATA estrato 0 | 4.128 | 16.0 | 23.189% | 24.8728 | +15.1219 | +155.08% |
| INT4 bloco 32 | 4.500 (nominal) | 4.5 | 9.961% | 11.3931 | +1.6422 | +16.84% |
| STRATA estrato 1 | 5.114 | 16.0 | 10.964% | 11.0266 | +1.2757 | +13.08% |
| INT5 bloco 32 | 5.500 (nominal) | 5.5 | 4.649% | 10.1046 | +0.3537 | +3.63% |
| STRATA estrato 2 | 6.114 | 16.0 | 5.385% | 10.0185 | +0.2676 | +2.74% |
| INT6 bloco 32 | 6.500 (nominal) | 6.5 | 2.250% | 9.8173 | +0.0664 | +0.68% |
| STRATA estrato 3 | 7.114 | 16.0 | 2.687% | 9.8108 | +0.0599 | +0.61% |
| STRATA estrato 4 | 8.114 | 16.0 | 1.357% | 9.7656 | +0.0147 | +0.15% |
| INT8 bloco 32 | 8.500 (nominal) | 8.5 | 0.549% | 9.7544 | +0.0035 | +0.04% |
| STRATA estrato 5 | 9.114 | 16.0 | 0.708% | 9.7564 | +0.0056 | +0.06% |
| STRATA estrato 6 | 10.114 | 16.0 | 0.408% | 9.7523 | +0.0014 | +0.01% |
| BF16 original (estrato 7) | 11.114 | 16.0 | 0.000% | 9.7509 | +0.0000 | +0.00% |

STRATA: disco medido com deflate-raw por plano (como o core); na RAM o corte volta a BF16 (16 bits), sem kernel próprio.
INT-k: disco e RAM nominais (k bits + escala FP16 a cada 32 pesos), sem medir compressão extra.
