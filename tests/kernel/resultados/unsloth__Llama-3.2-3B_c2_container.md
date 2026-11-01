# C1 + C2 no Llama-3.2-3B, no contêiner (CPU)

Regra escrita antes de rodar em [docs/10](../../../docs/10-motor-de-compressao.md), "O 3B no contêiner", 2026-10-31. Rodou
de 15:59 a 00:07 (8 h, 4 núcleos, 15 GB de RAM, sem GPU). Script: [c2_3b/b3.sh](c2_3b/b3.sh).

- **Modelo:** unsloth/Llama-3.2-3B (base), 28 camadas, 2,819 bilhões de pesos nas matrizes das camadas, embeddings
  amarrados à saída.
- **Nível 0:** Q4_K_M com imatrix (125 trechos da calibração misturada, a partir do BF16), intacto. **Níveis 1 e 2:**
  P e tabela linha (DL), bits do GPTQ (Hessianas de 500 trechos do `calib_big` numa passada, bf16; amortecimento 0,1).
  Erro de saída somado contra o nível 0: 0,229 e 0,069 (no 1B, 0,217 e 0,066).
- **KL de teste:** PyTorch, contra o modelo original, na segunda metade de 40 trechos de 512 tokens do wikitext-2 de teste
  e do C4 validação 00000. As camadas ficam em fp16 e as contas saem em fp32, uma camada por vez
  (`strata_dkl.py --fluxo --guarda fp16`). No 1B, esse modo deu a mesma KL do fp32 até a terceira casa: nível 1 8,416 contra
  8,416, Q4_K_M 38,471 contra 38,471 ([c2_3b/confere_fp16_1b.sh](c2_3b/confere_fp16_1b.sh)).
- **RAM** = bytes de todos os tensores, como no 1B. *Bits a mais* = quanto gasta além do k-quant de mesma KL; abaixo
  de 0, o STRATA ganha.

## C1: frente e níveis inteiros (KL ×1000)

| modelo | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 1918,4 | 23,89 ± 0,71 | | 23,35 ± 1,16 | |
| Q5_K_S | 2156,9 | 7,27 ± 0,16 | | 6,78 ± 0,19 | |
| Q5_K_M | 2207,1 | 8,46 ± 0,55 | | 8,74 ± 0,73 | |
| Q6_K | 2513,9 | 3,99 ± 0,30 | | 3,90 ± 0,29 | |
| Q8_0 | 3255,9 | 0,35 ± 0,01 | | 0,32 ± 0,01 | |
| nível 1 | 2254,4 | 5,55 ± 0,12 | −0,186 | 5,98 ± 0,25 | +0,048 |
| nível 2 | 2590,4 | 2,05 ± 0,04 | −0,375 | 2,32 ± 0,23 | −0,229 |

O Q5_K_M sai pior que o Q5_K_S, que é menor; fica fora da frente (no 1B aconteceu o mesmo).

**O 3B repete o atraso do 8B? Não.** O nível 1 no C4 deu +0,048 bit: entre o limite do 1B (≤ +0,04) e o do 8B (≥ +0,08),
uma faixa que a regra não nomeou. Ele fica bem mais perto do 1B (+0,04) que do 8B (+0,12 a +0,14). Então o 3B não serve
de banco de testes para o atraso do nível 1 no C4 do 8B. Pela tendência 1B → 3B → 8B, o 3B também promete demais.

## C2: misturas (KL ×1000)

Orçamentos: Q4_K_M + 0,35 / 0,55 / 0,75 do caminho entre o nível 0 e o nível 2.

| orçamento | alocador | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais | níveis inteiros (wiki / C4) | bits dos inteiros |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2,11 GiB | medido | 2150,9 | **8,28 ± 0,18** | **+0,060** | **8,52 ± 0,48** | **+0,113** | 8,70 / 9,10 | +0,090 / +0,151 |
| | de hoje | 2153,5 | 17,20 ± 1,13 | +0,504 | 15,35 ± 0,94 | +0,459 | | |
| 2,24 GiB | medido | 2284,4 | **4,70 ± 0,10** | **−0,392** | **5,10 ± 0,25** | **−0,169** | 5,08 / 5,50 | −0,255 / −0,024 |
| | de hoje | 2287,7 | 12,69 ± 0,98 | +0,722 | 11,29 ± 0,72 | +0,682 | | |
| 2,37 GiB | medido | 2419,4 | **3,18 ± 0,06** | **−0,487** | **3,56 ± 0,21** | **−0,362** | 3,40 / 3,76 | −0,425 / −0,314 |
| | de hoje | 2422,4 | 6,04 ± 0,15 | +0,461 | 6,62 ± 0,39 | +0,744 | | |

A soma prevista contra a medida, nos textos da medida: +0,4%, −1,3% e −7,2%.

**Veredito pela régua: sucesso.** O medido fica ≤ +0,10 bit além dos níveis inteiros nos dois textos (na verdade, abaixo
deles), ganha do de hoje nos três orçamentos e a soma prevê dentro de 10% com 28 camadas.

## Leitura

- **O C2 passou do 1B para o 3B sem perder força.** O ganho sobre os níveis inteiros foi de +0,03/+0,04 bit em 2,11 GiB,
  +0,14/+0,15 em 2,24 GiB e +0,06/+0,05 em 2,37 GiB (wiki / C4). No 1B tinha sido 0,06–0,12 perto do nível 1. A soma
  continua prevendo bem com quase o dobro de camadas, e o erro tem o mesmo jeito do 1B: subestima um pouco no orçamento
  mais alto.
- **O alocador de hoje continua quebrado com os níveis do GPTQ:** fica de +0,46 a +0,74 bit, pior até que cortar a
  escada sem alocador.
- **Abaixo do nível 1, o STRATA ainda perde para os k-quants:** em 2,11 GiB, +0,06 no wikitext e +0,11 no C4. É o
  trecho onde a frente sobe mais rápido (do Q4_K_M ao Q5_K_S, a KL cai 3,3 vezes em 238 MiB). O C2 encurta esse atraso,
  mas não fecha.
- **O que o plano escolhe:** de novo, o ffn_down das 4 últimas camadas fica no nível 0 em todos os orçamentos (só +0,04
  de KL para descer). O attn_q quase não pesa e é o primeiro a descer. Os mais sensíveis são attn_output no meio
  (camadas 8–15) e ffn_down das 4 primeiras.
- **O que o 3B não diz:** nada sobre o 8B, que ainda decide. O que ele diz é que o C2 não dependia de ter só 16 camadas.

Arquivos: [c2_3b/](c2_3b/) (ΔKL de cada grupo, planos, KLs de teste, `avalia.json`, `regua.md`, script da régua e
erro de saída do GPTQ por tensor).
