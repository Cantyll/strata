# Livro de códigos aninhado, tipos reais STRATA_LIVRO0/1: conferência no Llama-3.2-1B (container, CPU)

Modelo `unsloth/Llama-3.2-1B`, imatrix do wikitext-2 de treino (125 trechos), KL contra o BF16 nos 40 primeiros trechos
do wikitext-2 raw de teste (contexto 512), llama.cpp no commit 2a66703 com `kernel/ggml/strata-tipos.patch`, CPU de 4
threads (AVX-512 disponível, kernels do STRATA em AVX2). GGUFs feitos com
`strata_livro.py --modo bloco --bits 4 5 --formato --out e0.gguf e1.gguf --out-tipo t0.gguf t1.gguf` sobre o Q4_K_M com
imatrix (tensores de fora das camadas vêm dele).

## Os pesos

`tests/kernel/confere_livro_gguf.c`: cada tensor do livro desquantizado pelo ggml (to_float do tipo novo, tabela lida da
chave `strata.livro.<tensor>`) contra o GGUF F16 da emulação.

| nível | tensores | pesos diferentes depois do arredondamento F16 |
|---|---:|---:|
| 0 | 112 | 0 |
| 1 | 112 | 0 |

## A KL (×1000)

O caminho F16 desquantiza os pesos para F16 e não arredonda as ativações; o tipo real na CPU arredonda as ativações para
Q8_K (vale para todos os tipos quantizados do ggml na CPU). Os k-quants entram de controle.

| modelo | bits/peso | caminho F16 | tipo real na CPU | efeito das ativações em Q8_K |
|---|---:|---:|---:|---:|
| livro nível 0 (tabela int8) | 4,25 | 49,51 ± 0,85 | 51,78 ± 0,98 | +2,27 (+4,6%) |
| IQ4_XS imatrix (controle) | 4,27 | 47,68 ± 0,73 | 49,31 ± 0,77 | +1,63 (+3,4%) |
| livro nível 1 (tabela int8) | 5,25 | 17,94 ± 0,33 | 20,03 ± 0,34 | +2,09 (+11,6%) |
| Q5_K_S imatrix (controle) | 5,50 | 11,53 ± 0,21 | 13,74 ± 0,39 | +2,20 (+19,1%) |

O efeito das ativações tem o mesmo tamanho nos tipos do livro e nos de controle: o kernel lê o formato certo (um erro de
layout daria KL dezenas de vezes maior).

**Custo das tabelas em int8** (mesmo quantizador, livro em float, docs/09 triagem do 1B): erro ponderado 0,4718% → 0,4750%
no nível 0 (+0,7%) e 0,1573% → 0,1606% no nível 1 (+2,1%); KL 49,46 → 49,51 e 17,36 → 17,94 (+3,3%). Pela inclinação
da curva do próprio livro (~2,7× de KL por bit), o nível 1 perde ~0,02–0,03 bit/peso com o int8; o nível 0, quase nada.

## Velocidade na CPU (llama-bench, 4 threads, tok/s, rodadas alternadas)

| modelo | geração (tg64) | prompt (pp128) |
|---|---:|---:|
| livro nível 0 | 30,8 | 416 |
| IQ4_XS imatrix | 27,0 | 396 |
| livro nível 1 | 25,8 | 330 |
| Q5_K_S imatrix | 21,4 | 348 |

Geração: nível 0 1,14× o IQ4_XS, nível 1 1,20× o Q5_K_S. Prompt: 1,05× e 0,95×. O container varia 15–20% entre
rodadas; a vantagem na geração vem sobretudo do prefetch que os kernels do STRATA fazem e o do IQ4_XS não. A régua de
velocidade (docs/09) é na GPU, no notebook do Qwen3-8B-Base.
