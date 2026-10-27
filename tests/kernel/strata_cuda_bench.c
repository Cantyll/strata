// Mede na GPU, sem modelo, o tempo do MMVQ de uma coluna (o caminho da geração) por tipo, nas formas do FFN do
// Llama-3.1-8B (4096 → 14336): (a) uma matriz só (o up) e (b) gate + up + swiglu, que o llama.cpp funde num kernel
// quando os dois tipos permitem. L camadas independentes num grafo, blocos aleatórios; o grafo roda R vezes.
// Os tipos D rodam com uma tabela qualquer (kernel exato) e com uma tabela linha (as três classes iguais).
//
//   L=llama.cpp; cc -O2 tests/kernel/strata_cuda_bench.c -I $L/ggml/include -L $L/build/bin -lggml -lggml-base -lggml-cpu -lggml-cuda -lm -Wl,-rpath,$L/build/bin -o strata_cuda_bench
//   ./strata_cuda_bench [L] [R] [nome]     # GGML_CUDA_DISABLE_FUSION=1 desliga a fusão; nome escolhe um caso só
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static int8_t tab[16];   // tabela de sub-células dos tipos D: |δ| ≤ 93 e a marca no último byte

// blocos aleatórios com escalas pequenas (os mesmos dos tipos P para os D), como em strata_cuda_check.c
static void aleatorio(int type, uint8_t * x, size_t nb) {
    const size_t bs = ggml_type_size(type);
    if (ggml_strata_tipo_d(type)) type -= 4;
    const ggml_fp16_t d = ggml_fp32_to_fp16(0.003f), dm = ggml_fp32_to_fp16(0.002f);
    for (size_t i = 0; i < nb; i++) {
        uint8_t * b = x + i*bs;
        for (size_t j = 0; j < bs; j++) b[j] = rnd();
        if (type == 46) {
            for (int j = 0; j < 256; j++) b[j] = (uint8_t) (int8_t) ((int) (rnd() % 256) - 128);
            memcpy(b + 272, &d, 2);
        } else if (type == GGML_TYPE_Q6_K || type == 45) {
            for (int j = 0; j < 16; j++) b[192 + j] = (uint8_t) (int8_t) ((int) (rnd() % 255) - 127);
            memcpy(b + 208, &d, 2);
        } else {
            memcpy(b, &d, 2); memcpy(b + 2, &dm, 2);
        }
    }
}

// µs por camada: fundir = 0, só o up; fundir = 1, gate + up + swiglu
static double mede(ggml_backend_t gpu, int type, int L, int R, int fundir) {
    const int K = 4096, M = 14336;
    const int nt = 5*L + 8;   // x, e por camada 2 pesos, gate, up e swiglu
    struct ggml_init_params ip = { nt*ggml_tensor_overhead() + ggml_graph_overhead_custom(nt, false), NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    struct ggml_tensor ** w = malloc(sizeof(*w)*2*L);
    struct ggml_cgraph * g = ggml_new_graph_custom(ctx, nt, false);
    for (int l = 0; l < L; l++) {
        for (int j = 0; j < 2; j++) {
            w[2*l + j] = ggml_new_tensor_2d(ctx, type, K, M);
            if (ggml_strata_com_tab(type)) memcpy(w[2*l + j]->op_params, tab, sizeof(tab));
        }
        struct ggml_tensor * up = ggml_mul_mat(ctx, w[2*l + 1], x);
        if (fundir) {
            struct ggml_tensor * gate = ggml_mul_mat(ctx, w[2*l], x);
            ggml_build_forward_expand(g, ggml_swiglu_split(ctx, gate, up));
        } else {
            ggml_build_forward_expand(g, up);
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    const size_t nb = (size_t) K*M/256, bytes = nb*ggml_type_size(type);
    uint8_t * a = malloc(bytes);
    for (int i = 0; i < 2*L; i++) {
        aleatorio(type, a, nb);
        ggml_backend_tensor_set(w[i], a, 0, bytes);
    }
    float * xv = malloc(sizeof(float)*K);
    for (int i = 0; i < K; i++) xv[i] = ((int) (rnd() % 2001) - 1000) / 1000.0f;
    ggml_backend_tensor_set(x, xv, 0, sizeof(float)*K);
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(gpu, g);
    ggml_backend_synchronize(gpu);
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < R; i++) ggml_backend_graph_compute(gpu, g);
    ggml_backend_synchronize(gpu);
    const double us = (double) (ggml_time_us() - t0) / R / L;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(w); free(a); free(xv);
    return us;
}

int main(int argc, char ** argv) {
    const int L = argc > 1 ? atoi(argv[1]) : 16, R = argc > 2 ? atoi(argv[2]) : 50;
    const char * so = argc > 3 ? argv[3] : NULL;
    ggml_time_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { fprintf(stderr, "sem GPU CUDA\n"); return 2; }
    int8_t tab_qualquer[16];
    for (int i = 0; i < 15; i++) tab_qualquer[i] = (int8_t) ((int) (rnd() % 187) - 93);
    tab_qualquer[15] = 0x53;
    struct { const char * nome; int type; int linha; } casos[] = {
        { "q5_K", GGML_TYPE_Q5_K, 0 }, { "q6_K", GGML_TYPE_Q6_K, 0 },
        { "p1", 43, 0 }, { "d1", 47, 0 }, { "d1_linha", 47, 1 },
        { "p2", 44, 0 }, { "d2", 48, 0 }, { "d2_linha", 48, 1 },
        { "q6p1", 45, 0 }, { "q6d1", 49, 0 }, { "q6d1_linha", 49, 1 },
        { "q6p2", 46, 0 }, { "q6d2", 50, 0 }, { "q6d2_linha", 50, 1 },
    };
    printf("%-11s %12s %16s %8s   (µs; FFN 4096 → 14336, %d camadas, %d rodadas%s)\n", "tipo", "só o up", "gate+up+swiglu",
           "razão", L, R, getenv("GGML_CUDA_DISABLE_FUSION") ? ", sem fusão" : "");
    for (size_t c = 0; c < sizeof(casos)/sizeof(casos[0]); c++) {
        if (so && strcmp(so, casos[c].nome) != 0) continue;
        memcpy(tab, tab_qualquer, sizeof(tab));
        if (casos[c].linha) {   // a classe do meio copiada para as outras duas
            const int k = (casos[c].type == 47 || casos[c].type == 49) ? 1 : 2, gk = 1 << k;
            for (int r = 0; r < gk; r++) tab[r] = tab[r + 2*gk] = tab[r + gk];
        }
        const double u = mede(gpu, casos[c].type, L, R, 0), f = mede(gpu, casos[c].type, L, R, 1);
        printf("%-11s %12.2f %16.2f %8.3f\n", casos[c].nome, u, f, f / u);
        fflush(stdout);
    }
    ggml_backend_free(gpu);
    return 0;
}
