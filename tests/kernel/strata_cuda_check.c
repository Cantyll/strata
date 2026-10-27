// Confere os tipos do STRATA no CUDA do llama.cpp contra o CPU: multiplica matrizes de blocos montados por ativações
// float com N colunas (N ≤ 8 vai pelo produto escalar MMVQ; N maior, pela desquantização + cuBLAS) e compara com o
// produto em float64 dos pesos desquantizados. Os dois backends quantizam as ativações (q8_1 no CUDA, Q8_K no CPU),
// então nenhum é exato: o que importa é o CUDA errar tanto quanto o CPU. Um erro de layout dá erro perto de 1.
// Os tipos D (47 a 50) usam os blocos dos P com uma tabela de sub-células qualquer nos op_params do tensor; os do livro
// (51 e 52), blocos aleatórios com uma tabela de valores qualquer (16 ou 32 int8 e a marca no byte 32).
// Os tipos D rodam de novo com uma tabela "linha" (as três classes iguais): o CUDA troca para o kernel sem classe.
//
//   L=llama.cpp; cc -O2 tests/kernel/strata_cuda_check.c -I $L/ggml/include -L $L/build/bin -lggml -lggml-base -lggml-cpu -lggml-cuda -lm -Wl,-rpath,$L/build/bin -o strata_cuda_check
//   ./strata_cuda_check [blocos.bin tipo]          # sai com código 1 se algo divergir (tipos D do arquivo: tabela qualquer)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rs = 12345;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static int8_t tab[16];   // tabela de sub-células dos tipos D: |δ| ≤ 93 e a marca no último byte
static int8_t tab_livro[GGML_MAX_OP_PARAMS];   // tabela do livro: 32 valores de −127 a 127, marca no byte 32

static const int8_t * tab_de(int type) { return ggml_strata_tipo_livro(type) ? tab_livro : tab; }

static void aleatorio(int type, uint8_t * x, size_t nb) {
    const size_t bs = ggml_type_size(type);
    if (ggml_strata_tipo_d(type)) type -= 4;   // mesmos blocos dos tipos P
    const ggml_fp16_t d = ggml_fp32_to_fp16(0.003f), dm = ggml_fp32_to_fp16(0.002f);
    for (size_t i = 0; i < nb; i++) {
        uint8_t * b = x + i*bs;
        for (size_t j = 0; j < bs; j++) b[j] = rnd();
        if (type == 46) {
            for (int j = 0; j < 256; j++) b[j] = (uint8_t) (int8_t) ((int) (rnd() % 256) - 128);
            memcpy(b + 272, &d, 2);
        } else if (ggml_strata_tipo_livro(type)) {      // bloco do IQ4_XS: só o d no começo
            memcpy(b, &d, 2);
        } else if (type == GGML_TYPE_Q6_K || type == 45) {   // o Q6_K_P1 é o bloco Q6_K com um plano depois
            for (int j = 0; j < 16; j++) b[192 + j] = (uint8_t) (int8_t) ((int) (rnd() % 255) - 127);
            memcpy(b + 208, &d, 2);
        } else {
            memcpy(b, &d, 2); memcpy(b + 2, &dm, 2);
        }
    }
}

// C = A·B no backend; A: M linhas de K pesos do tipo, B: N colunas de K floats
static void mul(ggml_backend_t be, int type, int K, int M, int N, const void * a, const float * b, float * c) {
    struct ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, type, K, M);
    if (ggml_strata_tipo_d(type)) memcpy(ta->op_params, tab, sizeof(tab));
    if (ggml_strata_tipo_livro(type)) memcpy(ta->op_params, tab_livro, sizeof(tab_livro));
    struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    struct ggml_tensor * tc = ggml_mul_mat(ctx, ta, tb);
    struct ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, tc);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(ta, a, 0, ggml_nbytes(ta));
    ggml_backend_tensor_set(tb, b, 0, ggml_nbytes(tb));
    ggml_backend_graph_compute(be, g);
    ggml_backend_tensor_get(tc, c, 0, ggml_nbytes(tc));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static int confere(ggml_backend_t cpu, ggml_backend_t gpu, int type, int K, int M, const uint8_t * a) {
    static const int NS[] = { 1, 2, 5, 8, 9, 64, 512 };
    float * fa = malloc(sizeof(float)*K*M);
    if (ggml_strata_com_tab(type)) ggml_strata_tab_usar(tab_de(type));
    ggml_get_type_traits(type)->to_float(a, fa, (int64_t) K*M);
    int falhas = 0;
    for (size_t t = 0; t < sizeof(NS)/sizeof(NS[0]); t++) {
        const int N = NS[t];
        float * b = malloc(sizeof(float)*K*N), * c0 = malloc(sizeof(float)*M*N), * c1 = malloc(sizeof(float)*M*N);
        for (int i = 0; i < K*N; i++) b[i] = ((int) (rnd() % 2001) - 1000) / 1000.0f * ((i % K) % 97 == 0 ? 30.0f : 1.0f);
        mul(cpu, type, K, M, N, a, b, c0);
        mul(gpu, type, K, M, N, a, b, c1);
        double e0 = 0, e1 = 0, den = 0;
        for (int n = 0; n < N; n++) for (int m = 0; m < M; m++) {
            double r = 0;
            for (int k = 0; k < K; k++) r += (double) fa[(size_t) m*K + k] * b[(size_t) n*K + k];
            e0 += (c0[n*M + m] - r)*(c0[n*M + m] - r); e1 += (c1[n*M + m] - r)*(c1[n*M + m] - r); den += r*r;
        }
        e0 = sqrt(e0/den); e1 = sqrt(e1/den);
        const int ok = e1 < 3*e0 + 2e-3 && e1 < 1e-1;
        printf("  %-16s N=%-3d erro relativo: CPU %.2e  CUDA %.2e  %s\n", ggml_type_name(type), N, e0, e1, ok ? "OK" : "DIVERGE");
        falhas += !ok;
        free(b); free(c0); free(c1);
    }
    free(fa);
    return falhas;
}

int main(int argc, char ** argv) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { fprintf(stderr, "sem GPU CUDA\n"); return 2; }
    int falhas = 0;
    for (int i = 0; i < 15; i++) tab[i] = (int8_t) ((int) (rnd() % 187) - 93);
    tab[15] = 0x53;
    for (int i = 0; i < 32; i++) tab_livro[i] = (int8_t) ((int) (rnd() % 255) - 127);
    tab_livro[GGML_STRATA_LIVRO_MARCA_POS] = GGML_STRATA_LIVRO_MARCA;
    if (argc > 2) {
        const int type = atoi(argv[2]);
        FILE * f = fopen(argv[1], "rb");
        if (!f) { perror(argv[1]); return 2; }
        fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t * a = malloc(sz);
        if (fread(a, 1, sz, f) != (size_t) sz) return 2;
        fclose(f);
        const int K = 2048, M = (int) (sz / ggml_row_size(type, K));
        printf("%s: %d linhas de %d pesos do arquivo\n", argv[1], M, K);
        falhas += confere(cpu, gpu, type, K, M, a);
    } else {
        const int types[] = { GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, 43, 44, 45, 46, 47, 48, 49, 50, GGML_TYPE_IQ4_XS, 51, 52 };
        const int K = 4096, M = 512;
        for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
            const size_t nb = (size_t) K*M/256;
            uint8_t * a = malloc(nb*ggml_type_size(types[i]));
            aleatorio(types[i], a, nb);
            falhas += confere(cpu, gpu, types[i], K, M, a);
            if (ggml_strata_tipo_d(types[i])) {   // tabela linha: a classe do meio copiada para as outras duas
                int8_t salva[16];
                memcpy(salva, tab, sizeof(tab));
                const int k = (types[i] == 47 || types[i] == 49) ? 1 : 2, g = 1 << k;
                for (int r = 0; r < g; r++) tab[r] = tab[r + 2*g] = tab[r + g];
                printf("  (tabela linha)\n");
                falhas += confere(cpu, gpu, types[i], K, M, a);
                memcpy(tab, salva, sizeof(tab));
            }
            free(a);
        }
    }
    printf(falhas ? "%d casos divergem\n" : "tudo OK\n", falhas);
    return falhas ? 1 : 0;
}
