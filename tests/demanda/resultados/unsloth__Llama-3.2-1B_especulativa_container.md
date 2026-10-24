# E1: nível 0 contra nível 2 (Llama-3.2-1B, container)

`unsloth/Llama-3.2-1B`, Q4_K_M do `llama-quantize` sem imatrix, níveis 1 e 2 com os tipos P de `strata_gguf_niveis.py`,
llama.cpp `2a66703` com o patch de `kernel/ggml`. Referência = logits do **nível 2**, 60 trechos de 512 tokens do
wikitext-2 raw de teste (`llama-perplexity --kl-divergence`). Container: Xeon com AVX-512, 4 núcleos, 2026-10-23.
Script: [`e1_especulativa.sh`](../e1_especulativa.sh).

| modelo contra o nível 2 | KL ×1000 | mesmo 1º token (= aceitação do rascunho guloso) | RMS Δp |
|---|---:|---:|---:|
| nível 0 (Q4_K_M) | 47,1 ± 0,6 | **88,6 ± 0,3%** | 5,40% |
| nível 1 | 13,8 ± 0,2 | 93,9 ± 0,2% | 2,94% |
| Q6_K | 12,3 ± 0,2 | 94,2 ± 0,2% | 2,79% |

Geração (`llama-bench -p 0 -n 64 --repack 0`, tokens/s, 3 rodadas):

| | 1 thread | 4 threads |
|---|---:|---:|
| nível 0 (Q4_K_M do ggml, sem prefetch) | 6,47 ± 0,06 | 18,71 ± 0,45 |
| nível 2 (Q4_K_P2/Q6_K_P2, com prefetch) | 6,32 ± 0,38 | 19,57 ± 0,42 |

Teto da especulativa (`conta_demanda.py --alfa 0.886 --c 0.71 1.0`): com o custo pela memória (c = 0,71) o melhor é
**1,08×** (γ = 2); com o custo medido aqui (c ≈ 1, o kernel do Q4_K do ggml não tem o prefetch dos tipos do STRATA,
seção 29) fica **0,92×**, mais lento que o nível 2 sozinho. Decisão: a especulativa entre níveis sai do plano.
