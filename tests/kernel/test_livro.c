// Confere os tipos STRATA_LIVRO0 e STRATA_LIVRO1 (kernel/strata_q4kr.h): o produto escalar AVX2 contra a referência em C
// e a desquantização contra o produto escalar, em blocos e tabelas aleatórios. Também confere que o LIVRO1 com a tabela
// T1[2q + f] = T0[q] dá o mesmo que o LIVRO0 (o plano não muda nada se as duas metades da célula valem o mesmo).
//
//   cc -O2 -mavx2 -mfma -mf16c -I kernel tests/kernel/test_livro.c kernel/strata_q4kr.c kernel/strata_q4kr_avx2.c -lm -o test_livro
//   ./test_livro          # sai com código 1 se algo divergir
#include "strata_q4kr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int confere(const char * o_que, double a, double b, double tol) {
    if (fabs(a - b) <= tol * (1 + fabs(b))) return 0;
    printf("DIVERGE %s: %.9g contra %.9g\n", o_que, a, b);
    return 1;
}

int main(void) {
    const int cols = 4096, nb = cols / 256;
    int falhas = 0;
    srand(11);
    float * a = malloc(sizeof(float) * cols), * w = malloc(sizeof(float) * cols), * ay = malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; i++) a[i] = (float) rand() / RAND_MAX - 0.5f;
    strata_block_q8_K * y = malloc(sizeof(*y) * nb);
    strata_q8_K_quantize_row(a, y, cols);
    for (int b = 0; b < nb; b++) for (int i = 0; i < 256; i++) ay[256 * b + i] = y[b].d * y[b].qs[i];

    for (int k = 0; k <= 1; k++) for (int rep = 0; rep < 50; rep++) {
        const int bb = strata_livro_block_bytes(k);
        uint8_t * x = malloc((size_t) nb * bb);
        for (int i = 0; i < nb * bb; i++) x[i] = rand();
        for (int b = 0; b < nb; b++) { x[b * bb] = rand(); x[b * bb + 1] = 0x14 + rand() % 8; }   // d finito, de 1e-3 a 1e-2
        int8_t tab[32];
        for (int i = 0; i < 32; i++) tab[i] = (int8_t) (rand() % 255 - 127);
        if (rep == 0) for (int i = 0; i < 32; i++) tab[i] = i % 2 ? 127 : -127;   // extremos: o maddubs não pode saturar
        const float ref = strata_livro_vec_dot_q8_K(k, tab, x, y, cols);
        const float avx = strata_livro_vec_dot_q8_K_avx2(k, tab, x, y, cols);
        strata_livro_dequantize_row(k, tab, x, w, cols);
        double dq = 0;
        for (int i = 0; i < cols; i++) dq += (double) w[i] * ay[i];
        char nome[64];
        snprintf(nome, sizeof nome, "LIVRO%d rep %d AVX2", k, rep);
        falhas += confere(nome, avx, ref, 1e-5);
        snprintf(nome, sizeof nome, "LIVRO%d rep %d desquantização", k, rep);
        falhas += confere(nome, dq, ref, 1e-5);
        if (k == 1) {                         // o mesmo bloco lido como LIVRO0 com T0[q] = T1[2q] = T1[2q + 1]
            int8_t t0[16], t1[32];
            for (int q = 0; q < 16; q++) t0[q] = t1[2 * q] = t1[2 * q + 1] = tab[q];
            uint8_t * x0 = malloc((size_t) nb * STRATA_LIVRO0_BYTES);
            for (int b = 0; b < nb; b++) memcpy(x0 + b * STRATA_LIVRO0_BYTES, x + b * bb, STRATA_LIVRO0_BYTES);
            snprintf(nome, sizeof nome, "LIVRO1 = LIVRO0 rep %d", rep);
            falhas += confere(nome, strata_livro_vec_dot_q8_K_avx2(1, t1, x, y, cols), strata_livro_vec_dot_q8_K_avx2(0, t0, x0, y, cols), 1e-6);
            free(x0);
        }
        free(x);
    }
    printf(falhas ? "%d divergências\n" : "LIVRO0 e LIVRO1: AVX2 = referência = desquantização (100 casos)\n", falhas);
    return falhas != 0;
}
