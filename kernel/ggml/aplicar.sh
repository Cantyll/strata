#!/bin/sh
# Põe os tipos STRATA_Q4_K_P1/P2 e STRATA_Q6_K_P1/P2 (blocos montados na carga) e STRATA_Q4_K_D1/D2 e
# STRATA_Q6_K_D1/D2 (os mesmos blocos com a tabela de sub-células por tensor) e STRATA_LIVRO0/1 (livro de códigos
# aninhado, níveis 0 e 1: bloco do IQ4_XS com tabela int8 por tensor) num llama.cpp no commit 2a66703.
#   sh kernel/ggml/aplicar.sh PASTA_DO_LLAMA_CPP
#   cmake -B build -DGGML_NATIVE=ON && cmake --build build -j --target llama-bench llama-perplexity llama-cli
# GPU NVIDIA: o mesmo, com -DGGML_CUDA=ON (os .cuh de kernel/cuda vão para ggml/src/ggml-cuda).
set -e
K=$(cd "$(dirname "$0")/.." && pwd)
L=$1
[ -d "$L/ggml" ] || { echo "uso: $0 PASTA_DO_LLAMA_CPP" >&2; exit 1; }
git -C "$L" apply "$K/ggml/strata-tipos.patch"
mkdir -p "$L/ggml/src/strata" "$L/ggml/src/ggml-cpu/strata"
cp "$K/strata_q4kr.h" "$K/strata_q4kr.c" "$L/ggml/src/strata/"
cp "$K/strata_q4kr_avx2.c" "$K/strata_separado.h" "$K/strata_separado_avx2.c" "$L/ggml/src/ggml-cpu/strata/"
cp "$K"/cuda/strata-*.cuh "$L/ggml/src/ggml-cuda/"
echo "tipos do STRATA aplicados em $L"
