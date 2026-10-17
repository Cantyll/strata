Modelo `unsloth/Llama-3.2-1B`, 288627 tokens de teste, janelas de 2048, contas em float32. Sensibilidade de Fisher em 32752 tokens do wikitext-2 de treino.

| Configuração | Bits/peso no disco | PPL | ΔPPL % | Proxy de Fisher | Estratos (E0..E7) | Estrato médio q k v o gate up down |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| STRATA uniforme E1 | 4.594 | 11.0266 | +13.08% | 0.002355 | 0 112 0 0 0 0 0 0 | 1.0 1.0 1.0 1.0 1.0 1.0 1.0 |
| STRATA por tensor @ Q4_K_M (mín. E0) | 4.800 | 10.6197 | +8.91% | 0.001554 | 0 56 41 15 0 0 0 0 | 1.5 2.5 2.4 1.8 1.0 1.1 1.2 |
| STRATA uniforme E2 | 5.594 | 10.0185 | +2.74% | 0.0005765 | 0 0 112 0 0 0 0 0 | 2.0 2.0 2.0 2.0 2.0 2.0 2.0 |
| STRATA por tensor @ Q5_K_M (mín. E0) | 5.660 | 9.9749 | +2.30% | 0.0004588 | 0 4 63 40 5 0 0 0 | 2.0 3.2 3.1 2.6 1.8 2.0 2.2 |
| STRATA por tensor @ Q6_K (mín. E0) | 6.560 | 9.8081 | +0.59% | 0.0001318 | 0 0 8 61 35 8 0 0 | 3.0 4.3 4.2 3.6 2.5 3.0 3.1 |
| STRATA uniforme E3 | 6.594 | 9.8108 | +0.61% | 0.0001427 | 0 0 0 112 0 0 0 0 | 3.0 3.0 3.0 3.0 3.0 3.0 3.0 |
| BF16 original (estrato 7) | 10.594 | 9.7509 | +0.00% | 0 | 0 0 0 0 0 0 0 112 | 7.0 7.0 7.0 7.0 7.0 7.0 7.0 |

Bits no disco medidos como no strata_ppl.py (core v0.2), somando sinal+expoente e os planos 1..d_t de cada matriz. O mapa completo de estratos por matriz está no JSON. Referência da seção 13: Q4_K_M 4,80 b = +4,07%, Q5_K_M 5,66 b = +1,25%, Q6_K 6,56 b = +0,49%.
