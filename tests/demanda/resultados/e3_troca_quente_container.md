# E3: troca de nível com o modelo carregado (Llama-3.2-1B, container)

`unsloth/Llama-3.2-1B`. O modelo é carregado do Q4_K_M (`llama-quantize`, sem imatrix), mapeado do disco e sem
reempacotar os pesos (`use_extra_bufts = false`). As 112 matrizes das camadas sobem de nível com o layout misto do E2:
Q4_K → `STRATA_Q4_K_S1`/`S2`, onde a base continua sendo o arquivo mapeado e os planos ficam num buffer à parte, e
Q6_K → `STRATA_Q6_K_S1` no nível 1 e `STRATA_Q6_K_P2` (montado) no nível 2. llama.cpp `2a66703` com
[`kernel/ggml/aplicar.sh`](../../../kernel/ggml/aplicar.sh). Os planos vêm de um arquivo gerado por
`tests/kernel/strata_gguf_niveis.py --planos` (243 MB). Com a opção, os GGUFs n1/n2 continuam iguais byte a byte. O
programa é [`strata_subir.cpp`](../strata_subir.cpp). Container: Xeon 2,8 GHz com AVX-512, 4 núcleos virtuais,
máquina compartilhada, 2026-10-23.

```
python tests/kernel/strata_gguf_niveis.py --model m1b --gguf q4km.gguf --nivel 1 2 --out n1.gguf n2.gguf --planos planos.bin
./strata_subir conferir q4km.gguf planos.bin wiki.test.raw c -n 128      # 0 → 1 → 2 → 1 → 0, logits de cada posição
./strata_subir ref      n2.gguf   -          wiki.test.raw r_n2 -n 128   # o mesmo com o GGUF do nível carregado do zero
./strata_subir gerar    q4km.gguf planos.bin wiki.test.raw g.csv -n 300 --inicio 16 --t-subida 3 --nice 0
python tests/demanda/resumo_subida.py g.csv
```

## Resultado igual bit a bit

Os primeiros 128 tokens do wikitext-2 de teste, um token por `llama_decode`, como na geração. Os logits de todas as
posições (128 × 128 256 floats) foram comparados com `cmp`.

| depois de | comparado com | 3 rodadas |
|---|---|---|
| nada (nível 0) | `q4km.gguf` | iguais |
| subir para o nível 1 | `n1.gguf` (tipos P), carregado do zero | iguais |
| subir para o nível 2 | `n2.gguf` (tipos P), carregado do zero | iguais |
| descer para o nível 1 | `n1.gguf` | iguais |
| descer para o nível 0 | `q4km.gguf` | iguais |

## Velocidade na geração (4 threads, tokens/s, mesmas 128 posições)

| nível | trocado com o modelo rodando | GGUF do nível carregado do zero | razão |
|---|---|---|---:|
| 0 | 19,83 · 21,54 · 20,88 | 20,37 · 21,50 · 20,62 | (mesmo código) |
| 1 (S1) | 17,46 · 18,41 · 18,44 | 18,47 · 18,40 · 18,23 | **1,00** (medianas) |
| 2 (misto) | 15,88 · 16,99 · 17,02 | 16,82 · 17,03 · 15,92 | **1,01** |

Este é o tokens/s que o E2 deixou para depois. No modelo inteiro, o layout misto roda na velocidade dos tipos P.

## Pausa durante a subida

O modo `gerar` processa 32 tokens de prompt, gera 300 tokens gulosos e começa a subida no token 16. Uma thread de fundo
lê os planos, converte (ordem qh, ou montagem do Q6_K_P2) e enfileira. Antes de cada token, a geração aplica o que
estiver pronto, o que só troca tipo, ponteiros e passos do tensor. "Normal" é a mediana dos tokens antes da subida, no
nível 0 com 4 threads. "Pausa" é quanto o token mais lento durou além de um token normal.

| threads da geração durante a subida | prioridade da subida | pausa máxima (em tokens normais) | subida completa (nível 1 / nível 2) |
|---|---|---|---|
| 4 (a subida disputa os 4 núcleos) | normal | **8,8 · 12,7 · 13,8** | 0,40 / 1,62 s · 0,18 / 1,31 s · 0,88 / 1,48 s |
| 4 | `nice 10` | 7,9 | 0,80 / 6,55 s |
| 3 (um núcleo livre para a subida) | normal | **0,61 · 0,97 · 0,68** | 0,20 / 1,26 s · 0,26 / 1,33 s · 0,61 / 1,54 s |
| 3 | `nice 10` | 0,83 | 0,20 / 1,37 s |

- **A troca em si é de graça:** no máximo 16 µs para aplicar até 97 trocas antes de um token.
- **A pausa vem da disputa pelos núcleos.** As threads do ggml esperam umas pelas outras a cada operação. Se a thread
  de fundo tira uma delas do núcleo, a multiplicação inteira espera, e um token chegou a levar 14 tokens normais. A
  prioridade baixa (`nice`) não resolve.
- **Com um núcleo cedido à subida, a pausa fica abaixo de 1 token nas 4 rodadas, mas com folga curta:** a pior foi
  0,97. Enquanto a subida corre (~1,3 s, ~20 tokens no 1B), a geração anda ~40% mais devagar com 3 threads e volta a
  4 threads quando termina.

Resumos das rodadas (os CSV de cada token estão em [`e3_csv/`](e3_csv/); `g_tX` = threads durante a subida, `nY` = nice):

```
g_t3n0.csv: antes p50 50.4 ms | durante 19 tokens, p50 67.8, p99 76.2, máx 81.0 ms (máx = 1.61 tokens normais; pausa = 0.61) | depois p50 63.6 | trocas aplicadas 224 em 15 tokens, máx 11 µs | níveis completos: 1 = True, 2 = True
g_t3n0_2.csv: antes p50 46.0 ms | durante 20 tokens, p50 66.6, p99 80.3, máx 90.4 ms (máx = 1.97 tokens normais; pausa = 0.97) | depois p50 60.3 | trocas aplicadas 224 em 13 tokens, máx 6 µs | níveis completos: 1 = True, 2 = True
g_t3n0_3.csv: antes p50 47.5 ms | durante 20 tokens, p50 67.9, p99 78.8, máx 79.9 ms (máx = 1.68 tokens normais; pausa = 0.68) | depois p50 64.5 | trocas aplicadas 224 em 12 tokens, máx 14 µs | níveis completos: 1 = True, 2 = True
g_t3n10.csv: antes p50 50.1 ms | durante 20 tokens, p50 66.7, p99 91.1, máx 91.7 ms (máx = 1.83 tokens normais; pausa = 0.83) | depois p50 62.3 | trocas aplicadas 224 em 14 tokens, máx 5 µs | níveis completos: 1 = True, 2 = True
g_t4n0.csv: antes p50 52.1 ms | durante 7 tokens, p50 145.2, p99 400.7, máx 511.5 ms (máx = 9.82 tokens normais; pausa = 8.82) | depois p50 58.0 | trocas aplicadas 224 em 5 tokens, máx 11 µs | níveis completos: 1 = True, 2 = True
g_t4n0_2.csv: antes p50 48.1 ms | durante 7 tokens, p50 101.4, p99 327.6, máx 658.2 ms (máx = 13.69 tokens normais; pausa = 12.69) | depois p50 60.0 | trocas aplicadas 224 em 6 tokens, máx 10 µs | níveis completos: 1 = True, 2 = True
g_t4n0_3.csv: antes p50 47.1 ms | durante 6 tokens, p50 194.2, p99 299.6, máx 695.4 ms (máx = 14.77 tokens normais; pausa = 13.77) | depois p50 62.5 | trocas aplicadas 224 em 5 tokens, máx 12 µs | níveis completos: 1 = True, 2 = True
g_t4n10.csv: antes p50 42.5 ms | durante 96 tokens, p50 60.3, p99 341.9, máx 378.4 ms (máx = 8.90 tokens normais; pausa = 7.90) | depois p50 60.0 | trocas aplicadas 224 em 24 tokens, máx 16 µs | níveis completos: 1 = True, 2 = True
```

Subida síncrona (modo `conferir`, tudo de uma vez numa thread, sem gerar): nível 1 em 0,14–0,16 s; nível 2 em
1,1–2,3 s, a maior parte montando os Q6_K_P2 e lendo os planos. Os planos estavam no cache do sistema, e o tempo do
disco ou da rede é o E4.

## Limites

- **1B, 4 núcleos, máquina compartilhada.** No 8B a subida leva ~7× mais tempo, então o período com 3 threads também
  é ~7× mais longo. A pausa depende do tempo de cada operação, não do total, e precisa ser medida de novo.
- **Só CPU com AVX2.** Na GPU, a base teria de estar na VRAM e os tipos S não têm kernel CUDA.
- **A ordem de subida é a do arquivo**, não a de sensibilidade. Isso não muda a pausa, muda a qualidade a meio
  caminho.
- **Prompt longo (lotes) usa o caminho genérico** para os tipos S. O caminho em tiles do llama.cpp ainda não conhece
  os tipos S, e o processamento do prompt fica mais lento nos níveis 1 e 2 (não medido).
