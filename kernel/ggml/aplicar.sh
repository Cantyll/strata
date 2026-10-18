#!/bin/sh
# Põe os tipos STRATA_Q4_K_P1/P2 e STRATA_Q6_K_P1/P2 (blocos montados na carga) num llama.cpp no commit 2a66703.
#   sh kernel/ggml/aplicar.sh PASTA_DO_LLAMA_CPP
#   cmake -B build -DGGML_NATIVE=ON && cmake --build build -j --target llama-bench llama-perplexity llama-cli
set -e
K=$(cd "$(dirname "$0")/.." && pwd)
L=$1
[ -d "$L/ggml" ] || { echo "uso: $0 PASTA_DO_LLAMA_CPP" >&2; exit 1; }
git -C "$L" apply "$K/ggml/strata-tipos.patch"
mkdir -p "$L/ggml/src/strata" "$L/ggml/src/ggml-cpu/strata"
cp "$K/strata_q4kr.h" "$K/strata_q4kr.c" "$L/ggml/src/strata/"
cp "$K/strata_q4kr_avx2.c" "$L/ggml/src/ggml-cpu/strata/"
echo "tipos do STRATA aplicados em $L"
