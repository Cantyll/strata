# C2 no Llama-3.2-1B: nível de cada grupo de matrizes escolhido pelo efeito medido na saída (contêiner)

Triagem do C2 de [docs/10](../../../docs/10-motor-de-compressao.md), 2026-10-30. Níveis do fim do C1: nível 0 = Q4_K_M
com imatrix, intacto; nível 1 com pontos P e nível 2 com a tabela linha (DL), ambos com bits do GPTQ (amortecimento 0,1,
Hessianas de 500 trechos da calibração misturada; `strata_gptq_niveis.py --pontos P,DL`).

- **Medida** (`tests/kernel/strata_dkl.py mede`): o modelo do Hugging Face roda em fp32 na CPU com os pesos dos níveis
  trocados em memória. Parte de todas as matrizes no nível 1 e mede cada grupo (7 tipos × faixas de 4 camadas = 28)
  um nível abaixo e um acima. Textos da medida: 3 trechos de 512 tokens do wikitext-2 de validação e 3 do C4 validação
  00001 depois dos 2 milhões de caracteres (fora dos de teste e da calibração). 57 passes, 27 min.
- **Plano** (`strata_dkl.py plano`): programação dinâmica supondo que os efeitos somam, custo em unidades de 64 KiB.
- **Alocador de hoje** (`strata_mistura.py`): erro com imatrix por matriz, guloso, sobre os mesmos níveis.
- **KL de teste:** llama-perplexity no caminho F16, 40 trechos do wikitext-2 de teste e do C4 validação 00000, contra o
  BF16. *Bits a mais* contra a frente dos k-quants com imatrix em F16, como no C1; RAM = Q4_K_M (762,8 MiB nas
  camadas) + bytes acima do Q4_K_M.

## Níveis inteiros

| nível | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais |
|---:|---:|---:|---:|---:|---:|
| 0 | 762,8 | 38,73 | 0 | 38,68 | 0 |
| 1 | 878,8 | 8,52 ± 0,18 | −0,113 | 9,83 ± 0,62 | +0,036 |
| 2 | 994,8 | 3,05 ± 0,06 | −0,415 | 3,57 ± 0,14 | −0,274 |

## Misturas (KL ×1000)

"Níveis inteiros" = KL e bits interpolados entre os níveis inteiros vizinhos no mesmo tamanho (log da KL linear em
RAM): é o que se tem sem alocador nenhum, só cortando a escada no meio.

| orçamento | alocador | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais |
|---|---|---:|---:|---:|---:|---:|
| 850 MiB | medido | 840,8 | **12,69 ± 0,22** | **+0,036** | **13,71 ± 0,43** | **+0,058** |
| | de hoje | 842,4 | 22,91 ± 0,39 | +0,387 | 24,23 ± 0,89 | +0,409 |
| | níveis inteiros | 840,8 | 13,99 | +0,092 | 15,40 | +0,127 |
| 900 MiB | medido | 890,8 | **7,28 ± 0,14** | **−0,234** | **8,13 ± 0,34** | **−0,124** |
| | de hoje | 892,4 | 16,57 ± 0,32 | +0,634 | 17,28 ± 0,69 | +0,640 |
| | níveis inteiros | 890,8 | 7,66 | −0,162 | 8,85 | −0,006 |
| 950 MiB | medido | 940,3 | **4,78 ± 0,09** | **−0,374** | 5,79 ± 0,41 | −0,170 |
| | de hoje | 942,4 | 7,62 ± 0,13 | +0,276 | 8,64 ± 0,48 | +0,405 |
| | níveis inteiros | 940,3 | 4,94 | −0,335 | 5,74 | −0,180 |

Na primeira passada, a programação dinâmica arredondava o custo de cada grupo para cima em MiB e deixava até 23 MiB do
orçamento sem uso (0,816 GiB num orçamento de 0,830). Com 850 MiB, isso deu 15,85 / 16,35, ainda bem melhor que o de
hoje. Corrigido para 64 KiB (commit 5671158); a tabela acima é a da correção.

## A soma prevê a medida? (`strata_dkl.py avalia`, mesmos textos da medida, KL do PyTorch)

| plano | prevista | medida | erro |
|---|---:|---:|---:|
| 850 MiB | 9,879 | 9,710 | +1,7% |
| 900 MiB | 5,544 | 5,504 | +0,7% |
| 950 MiB | 3,138 | 3,384 | −7,3% |

A KL do PyTorch é ~20% menor que a do llama-perplexity (todas no nível 1: 6,76 contra 8,52; fp32 contra o modelo do HF,
e não contra o GGUF BF16), mas a ordem e as diferenças batem. Planos de hoje medidos no PyTorch: 17,29 / 12,33 / 5,37.

## Triagem pela régua (escrita antes de rodar)

- **O medido ganha do de hoje em ≥ 2 de 3 orçamentos, nos dois textos:** ganha nos 3, com 33 a 56% menos KL. **Passa.**
- **A soma prevista erra ≤ 10%:** +1,7%, +0,7%, −7,3%. **Passa.**

Vai ao 8B.

## Leitura

- **O alocador de hoje quebra com os níveis do GPTQ.** Ele fica pior até que cortar a escada sem alocador nenhum
  (+0,28 a +0,64 bit, contra −0,34 a +0,13 dos níveis inteiros). O GPTQ troca erro nos pesos por erro menor na saída, e o erro com imatrix
  só vê os pesos. Por isso "ganhar do de hoje" virou uma barra baixa, e a comparação que importa é contra os níveis
  inteiros.
- **Contra os níveis inteiros**, o medido ganha 0,06–0,07 bit em 850 MiB e 0,07–0,12 em 900 MiB, perto do nível 1. Em
  950 MiB empata (−0,04 no wikitext, +0,01 no C4, dentro da barra). Perto do nível 1 é onde o 8B ainda perde para o
  k-quant no C4 (+0,12 a +0,14): é ali que o C2 pode fechar a conta.
- **O que o plano escolhe:** as matrizes pequenas de atenção (k, v, output) sobem para o nível 2, que custa pouco. O
  ffn_down das 4 últimas camadas desce ao nível 0 (só +0,58 de KL), enquanto o das 4 primeiras é o mais sensível de
  todos (+1,53 para baixo, −0,68 para cima).
- **O que o 1B não diz:** se a soma continua prevendo bem com 32 camadas e se os ganhos se mantêm no 8B (o 1B já
  prometeu mais que o 8B entregou pelo menos quatro vezes).

Arquivos: [c2_1b/](c2_1b/) (`dkl.json` com o ΔKL de cada grupo, planos, `avalia.json`, `regua.json` e o script da
régua).
