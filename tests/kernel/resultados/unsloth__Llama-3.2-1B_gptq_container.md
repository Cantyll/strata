# C1 no Llama-3.2-1B: bits dos níveis escolhidos com a Hessiana (contêiner, só qualidade)

Triagem do C1 de [docs/10](../../../docs/10-motor-de-compressao.md). Nível 0 = Q4_K_M com imatrix, intacto. Pontos dos
filhos dos tipos P (meio de cada metade da célula). Muda só a escolha do bit: `perto` = filho mais próximo (o de hoje,
bit a bit igual aos tipos P); `gptq` = GPTQ com a Hessiana inteira das entradas de cada matriz, ordem por ativação, blocos
de 128 colunas. `tests/kernel/strata_hessiana.py` + `tests/kernel/strata_gptq_niveis.py`.

Caminho F16 (matrizes desquantizadas, nenhum lado arredonda as ativações). KL ×1000 contra o BF16 em 40 trechos de 512
tokens de dois textos: wikitext-2 raw de teste e C4 en (começo do arquivo de validação 00000, ~1,5 MB). A frente são os
k-quants com a mesma imatrix, também em F16.

**Bits a mais** = quanto o nível gasta além da frente de mesma KL (interpolação do log da KL entre os k-quants vizinhos
da frente), em bits por peso das matrizes das camadas (973 milhões). RAM do nível k = Q4_K_M (762,8 MiB) + k bits por
peso: nível 1 878,8 MiB, nível 2 994,8 MiB. Abaixo de 0 = o STRATA gasta menos que o k-quant.

## Calibração só com wikitext (imatrix e Hessianas: 125 trechos do texto de treino)

| modelo | nível | wikitext | bits a mais | C4 | bits a mais |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | | 38,55 | | 39,12 | |
| Q5_K_S | | 11,70 | | 11,60 | |
| Q5_K_M | | 11,79 | | 12,24 | |
| Q6_K | | 5,45 | | 5,55 | |
| Q8_0 | | 0,625 | | 0,670 | |
| P, perto (hoje) | 1 | 15,82 | +0,48 | 15,76 | +0,48 |
| P, perto (hoje) | 2 | 4,38 | −0,01 | 4,57 | +0,01 |
| P, gptq | 1 | 8,68 | −0,11 | 12,34 | +0,34 |
| P, gptq | 2 | 3,71 | −0,20 | **5,19** | **+0,16** |
| D, perto | 1 | 12,97 | +0,36 | | |
| D, gptq | 1 | 8,59 | −0,13 | | |
| D, gptq | 2 | 3,54 | −0,25 | | |

O GPTQ decorou o domínio da calibração: no C4, o nível 2 ficou **pior** que o de hoje. Os pontos D quase não somam nada
sobre o GPTQ (8,59 contra 8,68), então o tipo P, que já passa na velocidade, basta.

## Calibração misturada (imatrix e Hessianas iguais: 125 trechos, metade wikitext de treino, metade C4 validação 00001)

Os 340 mil primeiros caracteres de cada texto, em blocos alternados de 20 KB. O C4 de teste (00000) fica fora da
calibração. Os k-quants da frente são refeitos com essa imatrix.

| modelo | nível | wikitext | bits a mais | C4 | bits a mais |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | | 38,73 ± 0,73 | | 38,68 ± 1,23 | |
| Q5_K_S | | 11,39 ± 0,18 | | 11,91 ± 0,43 | |
| Q5_K_M | | 11,65 ± 0,22 | | 12,71 ± 0,63 | |
| Q6_K | | 5,42 ± 0,13 | | 5,54 ± 0,25 | |
| Q8_0 | | 0,625 ± 0,015 | | 0,670 ± 0,048 | |
| P, perto (hoje) | 1 | 16,71 ± 0,31 | +0,52 | 15,13 ± 0,44 | +0,44 |
| P, perto (hoje) | 2 | 4,56 ± 0,07 | +0,04 | 4,25 ± 0,10 | −0,07 |
| P, gptq, amortecimento 0,01 | 1 | 9,35 ± 0,21 | +0,02 | 10,38 ± 0,43 | +0,11 |
| P, gptq, amortecimento 0,01 | 2 | 3,74 ± 0,06 | −0,18 | 4,29 ± 0,14 | −0,06 |
| P, gptq, amortecimento 0,1 | 1 | 8,95 ± 0,25 | −0,04 | 10,70 ± 0,69 | +0,15 |
| P, gptq, amortecimento 0,1 | 2 | 3,28 ± 0,06 | −0,33 | 4,00 ± 0,25 | −0,14 |
| P, gptq, 0,1, Hessianas com 500 trechos | 1 | **8,52 ± 0,18** | **−0,11** | **9,83 ± 0,62** | **+0,04** |
| P, gptq, 0,1, Hessianas com 500 trechos | 2 | **3,17 ± 0,06** | **−0,37** | **3,55 ± 0,09** | **−0,28** |
| P, gptq, 0,1, Hessianas com 1000 trechos | 1 | 8,49 ± 0,16 | −0,12 | 9,69 ± 0,40 | +0,02 |
| P, gptq, 0,1, Hessianas com 1000 trechos | 2 | 3,10 ± 0,05 | −0,40 | 3,51 ± 0,10 | −0,29 |

## Pontos do tipo D do kernel com GPTQ (2026-10-29)

Mesma calibração, amortecimento 0,1 e Hessianas de 500 trechos. DC = D exato do kernel (tabela por classe do código do
nível 0 e caminho, `--pontos DC`); DL = tabela linha, uma classe só (`--pontos DL`, a variante rápida do PR #16).

| modelo | nível | wikitext | bits a mais | C4 | bits a mais |
|---|---:|---:|---:|---:|---:|
| P, gptq | 1 | 8,52 ± 0,18 | −0,11 | 9,83 ± 0,62 | +0,04 |
| DL, gptq | 1 | 9,17 ± 0,18 | −0,01 | 9,47 ± 0,31 | −0,02 |
| DC, gptq | 1 | 8,96 ± 0,17 | −0,04 | 9,31 ± 0,25 | −0,04 |
| P, gptq | 2 | 3,17 ± 0,06 | −0,37 | 3,55 ± 0,09 | −0,28 |
| DL, gptq | 2 | 3,17 ± 0,05 | −0,37 | 3,55 ± 0,19 | −0,28 |
| DC, gptq | 2 | 3,13 ± 0,06 | −0,39 | 3,36 ± 0,11 | −0,34 |

Com GPTQ, os pontos do D quase não somam no 1B. O linha empata com o P nos dois níveis; o D exato ganha ~5% de KL no C4.

Na última linha, só as Hessianas mudam: 500 trechos (256 mil tokens) da mesma receita, com 1,3 milhão de caracteres de
cada texto. A imatrix e a frente continuam as de 125 trechos.

Ganho sobre o P de hoje mantido no C4 (razão dos log da KL, C4 / wikitext): amortecimento 0,01, nível 1 65% e nível 2 −5%;
0,1, 55% e 18%; 0,1 com 500 trechos, **64% e 49%**; com 1000 trechos, 66% e 50%. De 500 para 1000 trechos o ganho
muda pouco (dentro da barra de erro): no 1B, a alavanca do texto de calibração saturou perto de 500.

## Leitura

- **Triagem: passa.** O GPTQ tira 46% da KL do nível 1 e 28% do nível 2 no wikitext (régua: 10% e 5%).
- **Nível 1:** o atraso cai de +0,52 para −0,04 bit no wikitext e de +0,44 para +0,15 no C4. É o maior ganho de qualquer
  experimento nos níveis desde o PR #9 (no 8B, dez rodadas de escala e pontos tiraram ~0,2 bit da soma dos dois).
- **Nível 2:** com 125 trechos, ganha bem no wikitext e pouco no C4 (4,25 → 4,00). Com 500 trechos de Hessiana, o ganho no
  C4 vai a 16% (4,25 → 3,55) e a retenção, de 18% para 49%. O que faltava era texto: as Hessianas do ffn_down têm 8192
  entradas, e 64 mil tokens eram poucos para estimar a matriz. No 1B o nível 2 já estava na frente, porque a frente entre
  Q6_K e Q8_0 é uma interpolação longa; no 8B ele está +0,20 bit atrás, e é lá que a retenção vai decidir.
- **Calibração importa mais que o amortecimento.** A mistura com C4 tirou o nível 2 do vermelho no C4, e 4× mais tokens
  quase dobraram o ganho fora do domínio. O amortecimento 10× maior ajudou o nível 2 nos dois textos e deixou o nível 1
  igual dentro do erro. Fica 0,1, com 500 trechos.
- **Nível 1 com 500 trechos:** −0,11 bit no wikitext e +0,04 no C4, dentro dos +0,05 da régua de sucesso nos dois textos.
- **O que o 1B não diz:** se o ganho sobrevive no 8B (o 1B já prometeu mais que o 8B entregou pelo menos quatro vezes) e
  se 256 mil tokens bastam para Hessianas de 14336 entradas (ffn_down do 8B). A imatrix da frente aqui é a de 125 trechos;
  no notebook do 8B, imatrix e Hessianas usam os mesmos 500.
