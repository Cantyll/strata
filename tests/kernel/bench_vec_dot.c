// Velocidade do produto escalar (matriz × vetor, em 1 e em N threads com OpenMP) dos tipos Q4_K_R<k>/Q6_K_R<k> em AVX2, contra o
// Q4_K, Q5_K e Q6_K do ggml no mesmo processador. Blocos aleatórios (só a velocidade importa aqui).
//
//   cc -O2 -march=native -ffp-contract=off -I kernel -I LLAMA/ggml/include tests/kernel/bench_vec_dot.c -fopenmp
//      kernel/strata_q4kr.c kernel/strata_q4kr_avx2.c -L LLAMA/build/bin -lggml-cpu -lggml-base -lm -o bench
//   LD_LIBRARY_PATH=LLAMA/build/bin ./bench [linhas] [colunas]
#include "strata_q4kr.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint8_t *random_blocks(size_t nblocks, int bb, int d_off) {
    uint8_t *x = malloc(nblocks * bb);
    for (size_t i = 0; i < nblocks * bb; i++) x[i] = (uint8_t)rand();
    for (size_t b = 0; b < nblocks; b++) {          // d (e dmin no Q4_K) = 1,0 em fp16, para não gerar NaN
        x[b * bb + d_off] = 0x00, x[b * bb + d_off + 1] = 0x3C;
        if (d_off == 0) x[b * bb + 2] = 0x00, x[b * bb + 3] = 0x3C;
    }
    return x;
}

typedef float (*strata_dot)(int, const uint8_t *, const strata_block_q8_K *, int64_t);

static double run_strata(strata_dot f, int k, const uint8_t *x, int bb, const strata_block_q8_K *y, int rows, int cols,
                         float *sink) {
    const size_t row_bytes = (size_t)cols / 256 * bb;
    double best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now();
        float s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (int r = 0; r < rows; r++) s += f(k, x + r * row_bytes, y, cols);
        double t = now() - t0;
        *sink += s;
        if (t < best) best = t;
    }
    return best;
}

static double run_ggml(enum ggml_type type, const uint8_t *x, const void *y, int rows, int cols, float *sink) {
    const struct ggml_type_traits_cpu *tr = ggml_get_type_traits_cpu(type);
    const size_t row_bytes = ggml_row_size(type, cols);
    double best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now();
        float s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (int r = 0; r < rows; r++) {
            float v;
            tr->vec_dot(cols, &v, 0, x + r * row_bytes, 0, y, 0, 1);
            s += v;
        }
        double t = now() - t0;
        *sink += s;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 4096, cols = argc > 2 ? atoi(argv[2]) : 4096;
    const size_t nb = (size_t)rows * cols / 256;
    ggml_cpu_init();
    float *act = malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; i++) act[i] = (float)rand() / RAND_MAX - 0.5f;
    strata_block_q8_K *y = malloc(sizeof(strata_block_q8_K) * cols / 256);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(act, y, cols);   // mesmo layout do block_q8_K do ggml
    float sink = 0;
    const double w = (double)rows * cols;
    printf("matriz %d x %d, %d thread(s); ns por peso (menor de 5) e GB/s lidos dos pesos\n\n", rows, cols, omp_get_max_threads());
    printf("%-10s %6s %8s %7s\n", "tipo", "bits", "ns/peso", "GB/s");
    struct { const char *name; enum ggml_type t; } gg[] = {{"Q4_K", GGML_TYPE_Q4_K}, {"Q5_K", GGML_TYPE_Q5_K}, {"Q6_K", GGML_TYPE_Q6_K}};
    for (int i = 0; i < 3; i++) {
        const size_t bytes = ggml_row_size(gg[i].t, cols) * rows;
        uint8_t *x = random_blocks(nb, (int)(bytes / nb), gg[i].t == GGML_TYPE_Q6_K ? 208 : 0);
        double t = run_ggml(gg[i].t, x, y, rows, cols, &sink);
        printf("%-10s %6.2f %8.3f %7.1f  (ggml)\n", gg[i].name, 8.0 * bytes / w, 1e9 * t / w, bytes / t / 1e9);
        free(x);
    }
    for (int q6 = 0; q6 < 2; q6++) {
        for (int k = 1; k <= 3; k++) {
            const int bb = q6 ? strata_q6kr_block_bytes(k) : strata_q4kr_block_bytes(k);
            uint8_t *x = random_blocks(nb, bb, q6 ? 208 : 0);
            double t = run_strata(q6 ? strata_q6kr_vec_dot_q8_K_avx2 : strata_q4kr_vec_dot_q8_K_avx2, k, x, bb, y, rows, cols, &sink);
            double ts = run_strata(q6 ? strata_q6kr_vec_dot_q8_K : strata_q4kr_vec_dot_q8_K, k, x, bb, y, rows / 8, cols, &sink) * 8;
            printf("%s_R%d %6.2f %8.3f %7.1f  (AVX2; escalar %.3f ns/peso)\n", q6 ? "Q6_K" : "Q4_K", k, 8.0 * bb / 256,
                   1e9 * t / w, (double)nb * bb / t / 1e9, 1e9 * ts / w);
            free(x);
        }
    }
    fprintf(stderr, "(%g)\n", sink);
    return 0;
}
