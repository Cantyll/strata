#!/bin/sh
# E1 da precisão sob demanda (docs/08): o nível 0 concorda o bastante com o nível 2 para a decodificação especulativa?
# Pré-requisitos (como em kernel/README.md, "Dentro do llama.cpp"): llama.cpp no commit 2a66703 com o patch de
# kernel/ggml, q4km.gguf (llama-quantize Q4_K_M do BF16, sem imatrix), n1.gguf e n2.gguf de
# tests/kernel/strata_gguf_niveis.py --nivel 1 2, q6k.gguf, e o wikitext-2 raw.
#   sh tests/demanda/e1_especulativa.sh PASTA_DO_LLAMA_CPP/build/bin wikitext-2-raw/wiki.test.raw
B=${1:-llama.cpp/build/bin}; F=${2:-wikitext-2-raw/wiki.test.raw}
# logits do nível 2 como referência; os outros modelos são medidos contra eles ("Same top p" = aceitação gulosa)
$B/llama-perplexity -m n2.gguf -f $F -c 512 --chunks 60 -t 4 --kl-divergence-base base_n2.kld > ppl_n2.log 2>&1
for m in q4km n1 q6k; do
  $B/llama-perplexity -m $m.gguf -f $F -c 512 --chunks 60 -t 4 --kl-divergence-base base_n2.kld --kl-divergence > kld_$m.log 2>&1
done
$B/llama-bench -m q4km.gguf,n2.gguf -p 0 -n 64 -t 1,4 -r 3 --repack 0 > bench.log 2>&1
grep -E "Mean +KLD|Same top p" kld_*.log
grep -E "tg64" bench.log
