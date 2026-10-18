# Kernel do STRATA: tipos Q4_K_R<k> e Q6_K_R<k>

Referência escalar em C99 dos tipos que rodam os níveis do modo residual (especificação B.8). Ela define o resultado;
as versões AVX2, NEON e CUDA que vierem depois precisam reproduzi-lo bit a bit.

- `strata_q4kr.h`: layout do bloco e fórmula.
- `strata_q4kr.c`: desquantização de uma linha, quantização das ativações em Q8_K e produto escalar inteiro.
- `strata_q4kr_avx2.c`: produto escalar em AVX2, com o mesmo float da referência, bit a bit.
- Velocidade contra o ggml: `tests/kernel/bench_vec_dot.c` (instruções no topo do arquivo).
- Teste num modelo real: `python tests/kernel/test_q4kr.py --model unsloth/Llama-3.2-1B --gguf Q4_K_M.gguf`.

## O bloco

| bytes | conteúdo |
|---|---|
| 0–143 | o bloco Q4_K do estrato 0, byte a byte |
| 144–175 | plano do nível 1: 1 bit por peso |
| 176–207 | plano do nível 2 |
| 208–239 | plano do nível 3 |

Q4_K_R1 tem 176 bytes por 256 pesos (5,5 bits/peso, o tamanho do Q5_K), Q4_K_R2 208 (6,5, o do Q6_K), Q4_K_R3 240 (7,5).
Subir um nível é acrescentar 32 bytes no fim de cada bloco; o que já está na memória não muda.

## A conta

Com `a = d·sc` e `m = dmin·mn` do sub-bloco, como no Q4_K, `r` = os k bits dos planos (nível 1 mais significativo):

    t = 2^(k+1)·q4 + 2r + 1 − 2^k        v = a · t / 2^(k+1) − m

`v` é o centro da sub-célula. O mínimo é o do Q4_K, sem requantizar (a perda do GGUF Q5_K vinha de requantizá-lo).
`t` cabe em int8 até k = 2 (−3…123), então o produto escalar contra ativações Q8_K é uma soma de inteiros por sub-bloco:

    dot = Σ_blocos d_x · ( d/2^(k+1) · Σ_j sc_j · Σ t·q8  −  dmin · Σ_j mn_j · Σ q8 )

## Q6_K_R<k>

Para as matrizes que o Q4_K_M guarda em Q6_K. Bloco = o Q6_K byte a byte (210 bytes: `ql[128]`, `qh[64]`, 16 escalas
int8, `d`) seguido dos mesmos k planos de 32 bytes: 242, 274 e 306 bytes (7,56, 8,56 e 9,56 bits/peso nessas matrizes).
Com `a = d·sc_j` (sub-bloco de 16 pesos, escala com sinal) e `q` de −32 a 31: `t = 2^(k+1)·q + 2r + 1 − 2^k`, `v = a·t/2^(k+1)`.
`r` conta no sentido de `q` crescente (valor decrescente quando `sc < 0`, metade dos sub-blocos). A soma inteira do bloco
vai em 64 bits: com k = 3 o pior caso passa de 2^31.

## Estado

- Conferido no Llama-3.2-1B, as 112 matrizes (96 Q4_K com 830 M pesos e 16 Q6_K com 143 M), k = 1, 2, 3: desquantização
  igual bit a bit à fórmula em float32, igual à emulação que deu os ΔPPL (seções 20 e 22 dos resultados; diferença
  abaixo de 1,3e-6 do passo, arredondamento float32 contra float64), produto escalar com erro relativo ~1e-8.
- Os blocos `Q4_K_R` do leitor em Rust (`blocos_q4kr`, arquivo 0.3) são iguais byte a byte aos que o teste monta.
- AVX2: o produto escalar dá o mesmo float da referência (conferido em `test_q4kr.py`). Velocidade numa CPU de
  nuvem compartilhada (matriz 4096 × 4096, uma thread, o menor de 25 tempos; ruidoso, só indicativo), em ns por peso:

  | ggml | | STRATA | |
  |---|---:|---|---:|
  | Q5_K | 0,042 | Q4_K_R1 | 0,054 |
  | Q6_K | 0,036 | Q4_K_R2 | 0,069 |
  | Q6_K | 0,036 | Q6_K_R1 | 0,067 |

  O nível 1 do modelo (85% Q4_K_R1, 15% Q6_K_R1) sai cerca de 1,35 vez mais lento que o Q5_K_M no produto escalar; o
  nível 2, cerca de 2 vezes mais lento que o Q6_K. Duas ideias medidas em protótipo e deixadas de fora: planos na
  ordem do `qh` do Q5_K (bit do peso 32j + l no byte l, bit j) não muda o nível 1 e ganha 4 a 11% nos níveis 2 e 3,
  pouco para mudar o formato; somar em vetor de float entre blocos, como o ggml, ganha uns 5% e perderia a igualdade
  bit a bit com a referência.
- Falta: os tipos dentro do ggml (`ggml_type`, `vec_dot`, `mul_mat`), NEON e CUDA, e medir tokens por segundo de verdade.
