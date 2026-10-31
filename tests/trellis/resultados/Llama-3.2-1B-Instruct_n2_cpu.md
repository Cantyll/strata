# N2: trellis (IQ*_KT) contra k-quants e i-quants na CPU, Llama-3.2-1B-Instruct

Rodado em 2026-10-31 no contêiner (Intel Xeon @ 2,1 GHz, 4 núcleos, AVX-512, sem GPU). Régua escrita antes no [docs/12, seção 10](../../../docs/12-quantizacao-sub4-e-codecs.md).

- Binário: [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) commit `13cffc006b82`, compilado para CPU (`-DGGML_NATIVE=ON`). Todos os tipos rodam no mesmo binário, inclusive os k-quants.
- Modelo: `Llama-3.2-1B-Instruct-f16.gguf` e `Llama-3.2-1B-Instruct.imatrix` do bartowski.
- Quantização: `llama-quantize --imatrix bart.imatrix --token-embedding-type q8_0`. O embedding (que no 1B também é a saída) fica em Q8_0 em todos os tipos, para que só o decoder mude.
- KL: `llama-perplexity` contra o F16, wikitext-2 teste, 60 trechos de 512 tokens. PPL do F16 nesses trechos: 14,54.
- Velocidade: `llama-bench -t 4 -p 512 -n 128 -r 3`. O contêiner é ruidoso; diferenças abaixo de ~5% não contam.
- Bits: "decoder" = matrizes das camadas (comparável com os artigos e com o STRATA); "arquivo" = tudo, com o embedding em Q8_0 (21% dos parâmetros do 1B).
- Scripts: [`run.sh`](../run.sh), [`bpw.py`](../bpw.py), [`tabela.py`](../tabela.py).

| tipo | b/peso decoder | b/peso arquivo | KL ×1000 | PPL | mesmo top-1 % | pp512 tok/s | tg128 tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ2_XXS | 2.124 | 3.481 | 1266.37 ± 8.97 | 49.933 | 49.15 | 572 ± 105 | 22.56 ± 1.21 |
| IQ2_KT | 2.313 | 3.629 | 656.08 ± 5.83 | 27.474 | 62.41 | 476 ± 20 | 25.56 ± 1.47 |
| IQ2_XS | 2.361 | 3.667 | 876.25 ± 6.98 | 34.800 | 56.63 | 558 ± 20 | 23.27 ± 1.34 |
| IQ2_M | 2.686 | 3.924 | 515.74 ± 4.85 | 24.546 | 66.14 | 555 ± 71 | 21.60 ± 1.05 |
| IQ3_XXS | 3.070 | 4.226 | 274.42 ± 2.97 | 18.883 | 74.63 | 504 ± 63 | 21.80 ± 0.32 |
| IQ3_KT | 3.181 | 4.313 | 163.61 ± 1.85 | 17.013 | 79.81 | 502 ± 54 | 22.89 ± 0.10 |
| IQ3_S | 3.456 | 4.529 | 143.42 ± 1.44 | 16.980 | 81.31 | 542 ± 58 | 16.06 ± 0.11 |
| IQ3_K | 3.456 | 4.529 | 126.54 ± 1.35 | 16.361 | 82.42 | 602 ± 8 | 27.23 ± 0.94 |
| Q3_K_M | 3.842 | 4.833 | 104.96 ± 1.27 | 16.042 | 83.36 | 469 ± 25 | 27.44 ± 0.23 |
| IQ4_KT | 4.038 | 4.988 | 49.76 ± 0.61 | 15.235 | 88.72 | 495 ± 82 | 17.74 ± 0.54 |
| IQ4_XS | 4.272 | 5.172 | 42.18 ± 0.45 | 15.104 | 89.50 | 471 ± 24 | 24.58 ± 1.24 |
| IQ4_KS | 4.284 | 5.181 | 38.59 ± 0.46 | 15.027 | 89.99 | 565 ± 54 | 25.70 ± 0.40 |
| IQ4_K | 4.517 | 5.365 | 29.23 ± 0.33 | 15.045 | 91.11 | 549 ± 55 | 25.83 ± 0.35 |
| Q4_K_S | 4.552 | 5.392 | 34.56 ± 0.40 | 15.113 | 89.95 | 401 ± 22 | 24.52 ± 0.57 |
| Q4_K_M | 4.811 | 5.596 | 30.53 ± 0.36 | 15.018 | 90.53 | 395 ± 33 | 23.54 ± 0.72 |
| Q5_K_S | 5.500 | 6.139 | 10.11 ± 0.12 | 14.613 | 94.59 | 427 ± 33 | 21.86 ± 0.35 |
| Q5_K_M | 5.665 | 6.269 | 8.40 ± 0.11 | 14.593 | 94.95 | 414 ± 11 | 21.28 ± 0.32 |
| Q6_K | 6.562 | 6.976 | 3.47 ± 0.05 | 14.587 | 96.76 | 389 ± 18 | 19.50 ± 0.55 |
| Q8_0 | 8.500 | 8.501 | 0.96 ± 0.01 | 14.557 | 98.13 | 500 ± 40 | 18.07 ± 0.60 |

Tempo de quantizar o 1B (4 núcleos): k-quants 12–38 s; i-quants 70–267 s; IQ2_KT 233 s, IQ3_KT 598 s, IQ4_KT 608 s.

## Mesmo KL, quantos bits a menos que a curva do llama.cpp oficial

Curva do llama.cpp oficial = k-quants + i-quants (IQ2_XXS … Q6_K). Interpolação em log(KL) entre os dois vizinhos da curva.

| tipo | bits | KL ×1000 | bits da curva no mesmo KL | diferença | domina algum ponto da curva? | tg128 contra o vizinho de mesmo KL |
|---|---:|---:|---:|---:|---|---:|
| IQ2_KT (trellis) | 2,313 | 656 | 2,539 | −0,23 | sim: menor e 25% menos KL que o IQ2_XS | 1,18× IQ2_M |
| IQ3_KT (trellis) | 3,181 | 164 | 3,376 | −0,20 | não | 1,43× IQ3_S |
| IQ4_KT (trellis) | 4,038 | 49,8 | 4,194 | −0,16 | não | **0,72× IQ4_XS, 0,75× Q4_K_M** |
| IQ3_K (iqk, sem trellis) | 3,456 | 127 | 3,612 | −0,16 | sim: mesmo tamanho do IQ3_S, 12% menos KL | 1,70× IQ3_S |
| IQ4_KS (iqk, sem trellis) | 4,284 | 38,6 | 4,397 | −0,11 | quase: +0,01 b e 9% menos KL que o IQ4_XS | 1,05× IQ4_XS |
| IQ4_K (iqk, sem trellis) | 4,517 | 29,2 | 4,838 | −0,32 | sim: 0,29 b menor e 4% menos KL que o Q4_K_M | 1,10× Q4_K_M |

## Veredito pela régua

A régua dizia: IQ4_KT passa se ganhar ≥ 0,3 b da curva no mesmo KL e gerar a ≥ 0,90× do Q4_K_M. Ganhou 0,16 b e gerou a 0,75×. **Reprova nas duas.** IQ3_KT e IQ2_KT também ficam abaixo de 0,3 b (0,20 e 0,23), embora sejam rápidos. Pela regra escrita, não vai para o 8B.

Fora da régua: o tipo que mais ganha na CPU é o IQ4_K, que não é trellis. Ele domina o Q4_K_M (0,29 b menor, KL menor, 1,10× mais rápido).

Limites: um modelo só, de 1B; o 1B já errou a escala dos ganhos antes (para mais). CPU com AVX-512; notebooks comuns têm só AVX2, e os kernels mudam. Os k-quants aqui rodam nos kernels do ik_llama.cpp, não nos do llama.cpp oficial.
