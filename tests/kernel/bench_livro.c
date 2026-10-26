// O que custa consultar a tabela do livro de códigos aninhado (docs/09) no produto escalar, contra os tipos do ggml.
// Só velocidade: blocos aleatórios, tabelas aleatórias. Um bloco de 256 pesos tem o formato do IQ4_XS (d F16, escala de
// 6 bits por 32 pesos, códigos de 4 bits) mais k planos de 1 bit (níveis 1 e 2: 168 e 200 bytes, 5,25 e 6,25 bits/peso).
// O código do nível k é 2·(código do nível k − 1) + bit, e o valor sai de uma tabela int8 de 2^(4+k) valores por tensor:
// a tabela é partida em 2^k tabelas de 16 (uma por combinação dos bits de cima) e cada uma é consultada com pshufb pelos
// 4 bits de baixo; os planos escolhem entre elas com blendv. Nível 0 = o kernel do IQ4_XS com a tabela vinda do tensor.
//
//   cc -O2 -march=native -I LLAMA/ggml/include tests/kernel/bench_livro.c -fopenmp -L LLAMA/build/bin \
//      -lggml-cpu -lggml-base -lm -Wl,-rpath,LLAMA/build/bin -o bench_livro
//   ./bench_livro [linhas] [colunas]
#include "ggml.h"
#include "ggml-cpu.h"
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } blk_q8K;    // = block_q8_K do ggml
typedef struct { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; } blk_l0;   // = block_iq4_xs

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int bloco_bytes(int k) { return (int)sizeof(blk_l0) + 32 * k; }

static inline int escala(const blk_l0 *b, int ib) {
    return (((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((b->scales_h >> 2 * ib) & 3) << 4)) - 32;
}

// referência escalar: Σ tabela[código] · escala · d · y
static float dot_ref(int k, int n, const uint8_t *x, const blk_q8K *y, const int8_t *tab) {
    const int bb = bloco_bytes(k);
    double s = 0;
    for (int i = 0; i < n / 256; i++) {
        const blk_l0 *b = (const blk_l0 *)(x + (size_t)i * bb);
        const uint8_t *pl = x + (size_t)i * bb + sizeof(blk_l0);
        const float d = ggml_fp16_to_fp32(b->d) * y[i].d;
        for (int ib = 0; ib < 8; ib++) {
            long acc = 0;
            for (int j = 0; j < 32; j++) {
                int q = j < 16 ? b->qs[16 * ib + j] & 0xf : b->qs[16 * ib + j - 16] >> 4;
                for (int p = 0; p < k; p++) q = 2 * q + ((pl[32 * p + 4 * ib + j / 8] >> (j % 8)) & 1);
                acc += tab[q] * y[i].qs[32 * ib + j];
            }
            s += (double)d * escala(b, ib) * acc;
        }
    }
    return (float)s;
}

// 32 bits → 32 bytes 0xFF/0x00 (o bytes_from_bits_32 do ggml)
static inline __m256i bits_bytes(const uint8_t *p) {
    uint32_t x32;
    memcpy(&x32, p, 4);
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf);
    v = _mm256_or_si256(v, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
    return _mm256_cmpeq_epi8(v, _mm256_set1_epi64x(-1));
}

static inline __m256i mul_add(__m256i v, __m256i y, int16_t sc) {
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(v, v), _mm256_sign_epi8(y, v));
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(sc));
}

#define KERNEL(K)                                                                                                         \
    static float dot_k##K(int n, const uint8_t *x, const blk_q8K *y, const __m256i *T) {                                \
        const int bb = bloco_bytes(K);                                                                                    \
        const __m128i m4 = _mm_set1_epi8(0xf);                                                                            \
        __m256 acc = _mm256_setzero_ps();                                                                                 \
        for (int i = 0; i < n / 256; i++) {                                                                               \
            const blk_l0 *b = (const blk_l0 *)(x + (size_t)i * bb);                                                       \
            const uint8_t *pl = x + (size_t)i * bb + sizeof(blk_l0);                                                     \
            __m256i sumi = _mm256_setzero_si256();                                                                        \
            int16_t sc[8];                                                                                                \
            for (int ib = 0; ib < 8; ib++) sc[ib] = (int16_t)escala(b, ib);                                               \
            for (int ib = 0; ib < 8; ib++) {                                                                              \
                const __m128i q = _mm_loadu_si128((const __m128i *)(b->qs + 16 * ib));                                    \
                const __m256i q4 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));      \
                __m256i v;                                                                                                \
                if (K == 0) {                                                                                             \
                    v = _mm256_shuffle_epi8(T[0], q4);                                                                    \
                } else if (K == 1) {                                                                                      \
                    const __m256i b1 = bits_bytes(pl + 4 * ib);                                                           \
                    v = _mm256_blendv_epi8(_mm256_shuffle_epi8(T[0], q4), _mm256_shuffle_epi8(T[1], q4), b1);             \
                } else {                                                                                                  \
                    const __m256i b1 = bits_bytes(pl + 4 * ib), b2 = bits_bytes(pl + 32 + 4 * ib);                        \
                    const __m256i lo = _mm256_blendv_epi8(_mm256_shuffle_epi8(T[0], q4), _mm256_shuffle_epi8(T[1], q4), b2); \
                    const __m256i hi = _mm256_blendv_epi8(_mm256_shuffle_epi8(T[2], q4), _mm256_shuffle_epi8(T[3], q4), b2); \
                    v = _mm256_blendv_epi8(lo, hi, b1);                                                                   \
                }                                                                                                         \
                const __m256i yv = _mm256_loadu_si256((const __m256i *)(y[i].qs + 32 * ib));                              \
                sumi = _mm256_add_epi32(sumi, mul_add(v, yv, sc[ib]));                                                                                     \
            }                                                                                                             \
            acc = _mm256_fmadd_ps(_mm256_set1_ps(ggml_fp16_to_fp32(b->d) * y[i].d), _mm256_cvtepi32_ps(sumi), acc);       \
        }                                                                                                                 \
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));                                \
        s = _mm_hadd_ps(s, s);                                                                                            \
        s = _mm_hadd_ps(s, s);                                                                                            \
        return _mm_cvtss_f32(s);                                                                                          \
    }
KERNEL(0)
KERNEL(1)
KERNEL(2)

// tabela de 2^(4+k) valores → 2^k tabelas de 16, uma para cada combinação dos k bits de baixo do código
static void parte_tabela(int k, const int8_t *tab, __m256i *T) {
    for (int m = 0; m < (1 << k); m++) {
        int8_t t16[16];
        for (int q = 0; q < 16; q++) t16[q] = tab[(q << k) | m];
        T[m] = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)t16));
    }
}

static uint8_t *blocos(size_t nblocos, int bb) {
    uint8_t *x = aligned_alloc(64, (nblocos * bb + 63) / 64 * 64);
    for (size_t i = 0; i < nblocos * bb; i++) x[i] = (uint8_t)rand();
    for (size_t b = 0; b < nblocos; b++) x[b * bb] = 0x00, x[b * bb + 1] = 0x3C;   // d = 1,0 em F16
    return x;
}

typedef float (*dot_livro)(int, const uint8_t *, const blk_q8K *, const __m256i *);

static double mede_livro(dot_livro f, const uint8_t *x, int bb, const blk_q8K *y, const __m256i *T, int rows, int cols, float *sink) {
    const size_t rb = (size_t)cols / 256 * bb;
    double best = 1e9;
    for (int rep = 0; rep < 7; rep++) {
        double t0 = now();
        float s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (int r = 0; r < rows; r++) s += f(cols, x + r * rb, y, T);
        double t = now() - t0;
        *sink += s;
        if (t < best) best = t;
    }
    return best;
}

static double mede_ggml(enum ggml_type type, const uint8_t *x, const void *y, int rows, int cols, float *sink) {
    const struct ggml_type_traits_cpu *tr = ggml_get_type_traits_cpu(type);
    const size_t rb = ggml_row_size(type, cols);
    double best = 1e9;
    for (int rep = 0; rep < 7; rep++) {
        double t0 = now();
        float s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
        for (int r = 0; r < rows; r++) {
            float v;
            tr->vec_dot(cols, &v, 0, x + r * rb, 0, y, 0, 1);
            s += v;
        }
        double t = now() - t0;
        *sink += s;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? atoi(argv[1]) : 16384, cols = argc > 2 ? atoi(argv[2]) : 4096;
    const int nb = cols / 256;
    ggml_cpu_init();
    float *yf = malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; i++) yf[i] = (float)rand() / RAND_MAX - 0.5f;
    blk_q8K *y = malloc(sizeof(blk_q8K) * nb);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(yf, y, cols);
    float sink = 0;

    // conferência contra a referência escalar
    int8_t tab[64];
    for (int i = 0; i < 64; i++) tab[i] = (int8_t)(rand() % 255 - 127);
    int ok = 1;
    for (int k = 0; k <= 2; k++) {
        __m256i T[4];
        int8_t tk[64];
        for (int i = 0; i < (16 << k); i++) tk[i] = tab[i];
        parte_tabela(k, tk, T);
        uint8_t *x = blocos(nb * 4, bloco_bytes(k));
        dot_livro f = k == 0 ? dot_k0 : k == 1 ? dot_k1 : dot_k2;
        for (int r = 0; r < 4; r++) {
            const float a = f(cols, x + (size_t)r * nb * bloco_bytes(k), y, T);
            const float b = dot_ref(k, cols, x + (size_t)r * nb * bloco_bytes(k), y, tk);
            if (fabsf(a - b) > 1e-4f * fabsf(b) + 1e-3f) ok = 0, printf("DIVERGE k=%d linha %d: %g contra %g\n", k, r, a, b);
        }
        free(x);
    }
    printf("conferência contra a referência escalar: %s\n", ok ? "OK" : "DIVERGE");

    const int threads[] = {1, omp_get_max_threads()};
    printf("%d × %d pesos; ns por peso (melhor de 7)\n\n| tipo | bits/peso | 1 thread | %d threads |\n|---|---:|---:|---:|\n",
           rows, cols, threads[1]);
    struct { const char *nome; enum ggml_type t; } gg[] = {{"Q4_K (ggml)", GGML_TYPE_Q4_K}, {"IQ4_XS (ggml)", GGML_TYPE_IQ4_XS},
                                                            {"Q5_K (ggml)", GGML_TYPE_Q5_K}, {"Q6_K (ggml)", GGML_TYPE_Q6_K},
                                                            {"Q8_0 (ggml)", GGML_TYPE_Q8_0}};
    void *y80 = malloc(ggml_row_size(GGML_TYPE_Q8_0, cols));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(yf, y80, cols);
    for (size_t g = 0; g < sizeof gg / sizeof gg[0]; g++) {
        const size_t rb = ggml_row_size(gg[g].t, cols);
        uint8_t *x = malloc(rb * rows);
        for (size_t i = 0; i < rb * rows; i++) x[i] = (uint8_t)rand();
        const size_t blk = rb / (cols / ggml_blck_size(gg[g].t));       // d (e dmin) = 1,0 em F16 para não gerar NaN
        for (size_t b = 0; b < rb * rows / blk; b++) {
            uint8_t *p = x + b * blk;
            if (gg[g].t == GGML_TYPE_Q6_K) p += blk - 2;
            p[0] = 0x00, p[1] = 0x3C;
            if (gg[g].t == GGML_TYPE_Q4_K || gg[g].t == GGML_TYPE_Q5_K) p[2] = 0x00, p[3] = 0x3C;
        }
        double t[2];
        for (int h = 0; h < 2; h++) {
            omp_set_num_threads(threads[h]);
            t[h] = mede_ggml(gg[g].t, x, gg[g].t == GGML_TYPE_Q8_0 ? y80 : (void *)y, rows, cols, &sink);
        }
        printf("| %s | %.2f | %.4f | %.4f |\n", gg[g].nome, 8.0 * rb / cols, 1e9 * t[0] / ((double)rows * cols),
               1e9 * t[1] / ((double)rows * cols));
        free(x);
    }
    for (int k = 0; k <= 2; k++) {
        __m256i T[4];
        parte_tabela(k, tab, T);
        uint8_t *x = blocos((size_t)rows * nb, bloco_bytes(k));
        dot_livro f = k == 0 ? dot_k0 : k == 1 ? dot_k1 : dot_k2;
        double t[2];
        for (int h = 0; h < 2; h++) {
            omp_set_num_threads(threads[h]);
            t[h] = mede_livro(f, x, bloco_bytes(k), y, T, rows, cols, &sink);
        }
        printf("| livro nível %d (tabela de %d) | %.2f | %.4f | %.4f |\n", k, 16 << k, 8.0 * bloco_bytes(k) / 256,
               1e9 * t[0] / ((double)rows * cols), 1e9 * t[1] / ((double)rows * cols));
        free(x);
    }
    fprintf(stderr, "(%g)\n", sink);
    return !ok;
}
