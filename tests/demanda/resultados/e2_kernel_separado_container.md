# E2: kernel com a base do GGUF e os planos separados (container)

Pergunta: o produto escalar que lê o bloco Q4_K/Q6_K do GGUF publicado num buffer e os planos dos níveis em outros
buffers (tipos S, [`kernel/strata_separado.h`](../../../kernel/strata_separado.h)) gera tão rápido quanto os blocos
montados na carga (tipos P)? Régua do plano: S ≥ 0,95 do P.

Microbenchmark [`bench_separado.c`](../bench_separado.c): matriz × vetor com 65 536 linhas × 4 096 colunas (151 MB de
Q4_K, bem maior que os 33 MB de L3) e com 1 024 linhas (cabe no cache). Pesos e planos aleatórios; mesma conta nas três
formas. As sete formas se revezam a cada repetição, e cada número é a mediana de 15 (1 thread) ou 21 (4 threads)
repetições. Container: Xeon 2,8 GHz com AVX-512, 4 núcleos virtuais, máquina compartilhada, 2026-10-23.

```
cc -O2 -march=native -ffp-contract=off -fopenmp -I kernel tests/demanda/bench_separado.c kernel/strata_separado_avx2.c \
   kernel/strata_q4kr.c kernel/strata_q4kr_avx2.c -lm -o bench_separado
OMP_NUM_THREADS=1 ./bench_separado 65536 4096 15
OMP_NUM_THREADS=4 ./bench_separado 65536 4096 21
```

## Conferência

S1 e S2 dão o mesmo float, bit a bit, que os kernels AVX2 de Q4_K_P e Q6_K_P em todas as 256 linhas de teste, e ficam a
menos de 1e-6 do RMS das linhas da referência escalar do tipo R. S0 (a base sozinha) bate com a referência escalar do
k-quant.

## Velocidade: P/S com a matriz fora do cache (acima de 1 = o separado é mais rápido)

Uma linha por rodada completa do programa. A variação entre rodadas de 4 threads (até 20%) é da máquina compartilhada;
o rodízio deixa a comparação dentro de cada rodada justa.

| grade | nível | 1 thread (2 rodadas) | mediana | 4 threads (3 rodadas) | mediana |
|---|---:|---|---:|---|---:|
| Q4_K | 1 | 1,074 · 1,069 | **1,07** | 0,976 · 1,195 · 1,018 | **1,02** |
| Q4_K | 2 | 1,074 · 1,094 | **1,08** | 0,886 · 1,005 · 1,003 | **1,00** |
| Q6_K | 1 | 1,005 · 0,961 | **0,98** | 0,968 · 1,106 · 0,966 | **0,97** |
| Q6_K | 2 | 0,797 · 0,689 | **0,74** | 0,815 · 0,684 · 0,749 | **0,75** |

ns por peso, rodada 1 de cada (as outras estão no fim):

| grade | forma | 1 thread frio | 1 thread quente | 4 threads frio | 4 threads quente |
|---|---|---:|---:|---:|---:|
| Q4_K | base sozinha (S0) | 0,0787 | 0,0857 | 0,0318 | 0,0353 |
| Q4_K | montado P1 | 0,1057 | 0,1026 | 0,0402 | 0,0405 |
| Q4_K | separado S1 | 0,0984 | 0,1067 | 0,0412 | 0,0426 |
| Q4_K | intercalado R1 | 0,1315 | 0,1317 | 0,0496 | 0,0465 |
| Q4_K | montado P2 | 0,1340 | 0,1281 | 0,0476 | 0,0557 |
| Q4_K | separado S2 | 0,1248 | 0,1295 | 0,0537 | 0,0666 |
| Q4_K | intercalado R2 | 0,1522 | 0,1506 | 0,0603 | 0,0725 |
| Q6_K | base sozinha (S0) | 0,1825 | 0,1794 | 0,0491 | 0,0594 |
| Q6_K | montado P1 | 0,2042 | 0,1980 | 0,0548 | 0,0695 |
| Q6_K | separado S1 | 0,2031 | 0,1991 | 0,0566 | 0,0671 |
| Q6_K | intercalado R1 | 0,2201 | 0,2239 | 0,0609 | 0,0678 |
| Q6_K | montado P2 | 0,1733 | 0,1657 | 0,0495 | 0,0538 |
| Q6_K | separado S2 | 0,2174 | 0,2114 | 0,0607 | 0,0698 |
| Q6_K | intercalado R2 | 0,2655 | 0,2511 | 0,0737 | 0,0819 |

Por que o Q6_K no nível 2 não chega: o Q6_K_P2 guarda o peso pronto em 8 bits (w = 4·q + r) e o kernel só carrega e
multiplica; ele é mais rápido até que o Q6_K sozinho (0,173 contra 0,183 ns/peso numa thread). O separado precisa
desmontar o Q6_K (4 + 2 bits) e ainda juntar dois planos, então o teto dele é o próprio Q6_K, abaixo do P2.

## No modelo inteiro (soma ponderada pelos pesos de cada grade)

Q4_K_M do 1B: 67,2% dos pesos em Q4_K, 32,8% em Q6_K (contado no GGUF). 8B: ~78% / ~22%, **estimado** pela regra do
`llama-quantize` (`attn_v` e `ffn_down` em Q6_K em 16 das 32 camadas, mais o `output`), não contado. "Misto" = tensores
Q4_K no layout separado e tensores Q6_K montados (P) ao subir de nível. Mediana das rodadas, P/S:

| | 1B, 1 thread | 1B, 4 threads | 8B, 1 thread | 8B, 4 threads |
|---|---:|---:|---:|---:|
| nível 1, tudo separado | 1,03 | 0,99 | 1,04 | 1,00 |
| nível 2, tudo separado | 0,92 | **0,86** | 0,96 | **0,89** |
| nível 2, misto | 1,05 | 1,00 | 1,06 | 1,00 |

## Custo de subir de nível (1 thread)

| operação | ns por peso | 8B inteiro (~7,0 bi de pesos nas matrizes) |
|---|---:|---|
| converter um plano para a ordem qh (tipos S) | 0,036–0,058 | ~0,3 s por nível |
| montar Q6_K_P1 (anexar o plano) | 0,16–0,19 | |
| montar Q6_K_P2 | 5,7–5,9 | só os ~1,5 bi de pesos Q6_K: ~9 s |
| montar Q4_K_P1 / P2 | 5,5–6,9 | (não é necessário no misto) |

A montagem é escalar e roda fora da geração; com 4 threads cai para ~2–3 s no 8B.

## Saída completa das rodadas

`e2_t1`:
```
1 thread(s); colunas 4096; ns por peso (mediana de 15, formas em rodízio)
grade  nível forma                       frio    quente  P/S frio
Q4_K   0     base sozinha (S0)         0.0787    0.0857
Q4_K   1     montado P1                0.1057    0.1026
Q4_K   1     separado S1               0.0984    0.1067     1.074
Q4_K   1     intercalado R1            0.1315    0.1317
Q4_K   2     montado P2                0.1340    0.1281
Q4_K   2     separado S2               0.1248    0.1295     1.074
Q4_K   2     intercalado R2            0.1522    0.1506
Q6_K   0     base sozinha (S0)         0.1825    0.1794
Q6_K   1     montado P1                0.2042    0.1980
Q6_K   1     separado S1               0.2031    0.1991     1.006
Q6_K   1     intercalado R1            0.2201    0.2239
Q6_K   2     montado P2                0.1733    0.1657
Q6_K   2     separado S2               0.2174    0.2114     0.797
Q6_K   2     intercalado R2            0.2655    0.2511
conversão de um plano para a ordem qh: 0.0364 ns por peso, 1 thread (33.6 MB por plano)
```

`e2_t1_2`:
```
1 thread(s); colunas 4096; ns por peso (mediana de 15, formas em rodízio)
grade  nível forma                       frio    quente  P/S frio
Q4_K   0     base sozinha (S0)         0.0882    0.0887
Q4_K   1     montado P1                0.1107    0.1077
Q4_K   1     separado S1               0.1036    0.1009     1.068
Q4_K   1     intercalado R1            0.1356    0.1405
Q4_K   2     montado P2                0.1378    0.1356
Q4_K   2     separado S2               0.1260    0.1359     1.094
Q4_K   2     intercalado R2            0.1628    0.1564
Q6_K   0     base sozinha (S0)         0.2184    0.1953
Q6_K   1     montado P1                0.2080    0.1930
Q6_K   1     separado S1               0.2165    0.2038     0.961
Q6_K   1     intercalado R1            0.2334    0.2310
Q6_K   2     montado P2                0.1792    0.1720
Q6_K   2     separado S2               0.2600    0.2234     0.689
Q6_K   2     intercalado R2            0.3138    0.3067
conversão de um plano para a ordem qh: 0.0364 ns por peso, 1 thread (33.6 MB por plano)
montagem P1 do Q4_K: 6.0558 ns por peso, 1 thread
montagem P2 do Q4_K: 6.2229 ns por peso, 1 thread
montagem P1 do Q6_K: 0.1606 ns por peso, 1 thread
montagem P2 do Q6_K: 5.8782 ns por peso, 1 thread
```

`e2_t4`:
```
4 thread(s); colunas 4096; ns por peso (mediana de 21, formas em rodízio)
grade  nível forma                       frio    quente  P/S frio
Q4_K   0     base sozinha (S0)         0.0318    0.0353
Q4_K   1     montado P1                0.0402    0.0405
Q4_K   1     separado S1               0.0412    0.0426     0.976
Q4_K   1     intercalado R1            0.0496    0.0465
Q4_K   2     montado P2                0.0476    0.0557
Q4_K   2     separado S2               0.0537    0.0666     0.887
Q4_K   2     intercalado R2            0.0603    0.0725
Q6_K   0     base sozinha (S0)         0.0491    0.0594
Q6_K   1     montado P1                0.0548    0.0695
Q6_K   1     separado S1               0.0566    0.0671     0.968
Q6_K   1     intercalado R1            0.0609    0.0678
Q6_K   2     montado P2                0.0495    0.0538
Q6_K   2     separado S2               0.0607    0.0698     0.815
Q6_K   2     intercalado R2            0.0737    0.0819
conversão de um plano para a ordem qh: 0.0576 ns por peso, 1 thread (33.6 MB por plano)
```

`e2_t4_2`:
```
4 thread(s); colunas 4096; ns por peso (mediana de 21, formas em rodízio)
grade  nível forma                       frio    quente  P/S frio
Q4_K   0     base sozinha (S0)         0.0229    0.0299
Q4_K   1     montado P1                0.0331    0.0379
Q4_K   1     separado S1               0.0277    0.0366     1.196
Q4_K   1     intercalado R1            0.0416    0.0478
Q4_K   2     montado P2                0.0427    0.0488
Q4_K   2     separado S2               0.0425    0.0506     1.004
Q4_K   2     intercalado R2            0.0439    0.0509
Q6_K   0     base sozinha (S0)         0.0703    0.0805
Q6_K   1     montado P1                0.0898    0.1012
Q6_K   1     separado S1               0.0812    0.0796     1.105
Q6_K   1     intercalado R1            0.0966    0.1113
Q6_K   2     montado P2                0.0612    0.0626
Q6_K   2     separado S2               0.0895    0.1014     0.683
Q6_K   2     intercalado R2            0.0935    0.1082
conversão de um plano para a ordem qh: 0.0389 ns por peso, 1 thread (33.6 MB por plano)
montagem P1 do Q4_K: 5.7257 ns por peso, 1 thread
montagem P2 do Q4_K: 6.8926 ns por peso, 1 thread
montagem P1 do Q6_K: 0.1725 ns por peso, 1 thread
montagem P2 do Q6_K: 5.7744 ns por peso, 1 thread
```

`e2_t4_3`:
```
4 thread(s); colunas 4096; ns por peso (mediana de 21, formas em rodízio)
grade  nível forma                       frio    quente  P/S frio
Q4_K   0     base sozinha (S0)         0.0334    0.0401
Q4_K   1     montado P1                0.0393    0.0405
Q4_K   1     separado S1               0.0386    0.0430     1.018
Q4_K   1     intercalado R1            0.0526    0.0665
Q4_K   2     montado P2                0.0575    0.0610
Q4_K   2     separado S2               0.0573    0.0601     1.002
Q4_K   2     intercalado R2            0.0619    0.0682
Q6_K   0     base sozinha (S0)         0.0687    0.0808
Q6_K   1     montado P1                0.0883    0.0945
Q6_K   1     separado S1               0.0914    0.0908     0.967
Q6_K   1     intercalado R1            0.0947    0.1108
Q6_K   2     montado P2                0.0680    0.0708
Q6_K   2     separado S2               0.0908    0.1123     0.748
Q6_K   2     intercalado R2            0.0969    0.1203
conversão de um plano para a ordem qh: 0.0414 ns por peso, 1 thread (33.6 MB por plano)
montagem P1 do Q4_K: 5.4807 ns por peso, 1 thread
montagem P2 do Q4_K: 6.2482 ns por peso, 1 thread
montagem P1 do Q6_K: 0.1886 ns por peso, 1 thread
montagem P2 do Q6_K: 5.7234 ns por peso, 1 thread
```
