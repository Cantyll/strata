#!/bin/sh
# Compila e roda tests/kernel/strata_cuda_host.cpp: os kernels CUDA do STRATA conferidos na CPU, sem GPU.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
for f in "$R"/kernel/cuda/*.cuh; do sed 's/<<<[^>]*>>>//' "$f" > "$T/$(basename "$f")"; done
cc -std=c99 -O2 -c -o "$T/q4kr.o" "$R/kernel/strata_q4kr.c" -I"$R/kernel"
c++ -std=c++17 -O2 -o "$T/strata_cuda_host" "$R/tests/kernel/strata_cuda_host.cpp" "$T/q4kr.o" \
    -I"$R/tests/kernel/cuda_shim" -I"$T" -I"$R/kernel" -lm
"$T/strata_cuda_host" "$@"
