# Base fina contra k-quants com imatrix (Llama-3.2-1B, container)

CPU do container (4 núcleos), `llama-perplexity` do llama.cpp com o patch, KL contra o BF16 nos 40 primeiros trechos de 512
tokens do wikitext-2 raw de teste. imatrix: `llama-imatrix`, 125 trechos do wikitext-2 raw de treino. k-quants do
`llama-quantize`. STRATA "sobre o Q4_K_M" = `strata_gguf_niveis.py` (células do Q4_K subdivididas); base fina =
`strata_base_fina.py` (grade de 4 + T bits ajustada com a imatrix, Q4 por corte).

| modelo | nível | GGUF (GiB) | KL (×1000) | mesmo 1º token |
|---|---|---:|---:|---:|
| Q3_K_M com imatrix | | 0,643 | 130,10 ± 2,04 | 82,51% |
| Q4_K_M sem imatrix | | 0,752 | 47,74 ± 0,72 | 88,54% |
| Q4_K_M com imatrix | | 0,752 | 40,94 ± 0,67 | 89,69% |
| base fina T=1 | 0 | 0,752 | 54,20 ± 0,84 | 88,26% |
| base fina T=2 | 0 | 0,752 | 61,72 ± 1,14 | 87,60% |
| base fina T=2, pesos 1, 2, 4 | 0 | 0,752 | 55,40 ± 1,03 | 88,21% |
| base fina T=2, pesos 1, 4, 16 | 0 | 0,752 | 55,67 ± 0,97 | 88,03% |
| Q5_K_M sem imatrix | | 0,849 | 16,22 ± 0,25 | 93,06% |
| Q5_K_M com imatrix | | 0,849 | 13,87 ± 0,31 | 93,96% |
| STRATA sobre o Q4_K_M sem imatrix | 1 | 0,882 | 16,16 ± 0,26 | 93,50% |
| STRATA sobre o Q4_K_M com imatrix | 1 | 0,882 | 17,75 ± 0,27 | 93,15% |
| base fina T=1 | 1 | 0,882 | **12,45 ± 0,41** | 93,90% |
| base fina T=2 | 1 | 0,882 | 15,58 ± 0,26 | 93,28% |
| base fina T=2, pesos 1, 2, 4 | 1 | 0,882 | 14,27 ± 0,25 | 93,53% |
| base fina T=2, pesos 1, 4, 16 | 1 | 0,882 | 14,65 ± 0,25 | 93,49% |
| Q6_K sem imatrix | | 0,952 | 8,37 ± 0,14 | 94,77% |
| Q6_K com imatrix | | 0,952 | 7,28 ± 0,16 | 95,55% |
| STRATA sobre o Q4_K_M sem imatrix | 2 | 0,979 | 6,19 ± 0,12 | 95,61% |
| STRATA sobre o Q4_K_M com imatrix | 2 | 0,979 | 6,40 ± 0,10 | 95,61% |
| base fina T=2 | 2 | 0,979 | **5,29 ± 0,14** | 95,97% |
| base fina T=2, pesos 1, 2, 4 | 2 | 0,979 | 5,39 ± 0,08 | 96,07% |
| base fina T=2, pesos 1, 4, 16 | 2 | 0,979 | 5,55 ± 0,11 | 95,85% |
