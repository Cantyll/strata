// Confere os tipos D (sub-células por tensor, kernel/strata_q4kr.h): o produto escalar AVX2 contra a referência em C, a
// desquantização contra o produto escalar, e a tabela padrão contra os tipos P, em blocos e tabelas aleatórios. Também
// confere o Q6_K_P1 (que é o bloco Q6_K_R1) contra o Q6_K_R1 e o produto escalar AVX2 dos tipos P.
//
//   cc -O2 -mavx2 -mfma -mf16c -I kernel tests/kernel/test_q4kd.c kernel/strata_q4kr.c kernel/strata_q4kr_avx2.c -lm -o test_q4kd
//   ./test_q4kd          # sai com código 1 se algo divergir
#include "strata_q4kr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const int cols = 4096, nb = cols/256;
    int falhas = 0;
    srand(7);
    float * a = malloc(sizeof(float)*cols);
    for (int i = 0; i < cols; i++) a[i] = (float) rand()/RAND_MAX - 0.5f;
    strata_block_q8_K * y = malloc(sizeof(*y)*nb);
    strata_q8_K_quantize_row(a, y, cols);

    for (int q6 = 0; q6 < 2; q6++) for (int k = 1; k <= 2; k++) for (int rep = 0; rep < 20; rep++) {
        const int bb = q6 ? strata_q6kp_block_bytes(k) : strata_q4kp_block_bytes(k);
        uint8_t * x = malloc((size_t) nb*bb);
        for (int i = 0; i < nb*bb; i++) x[i] = rand();
        for (int b = 0; b < nb; b++) {             // escalas pequenas e, no Q6, códigos dentro da faixa de k bits
            uint8_t * p = x + b*bb;
            if (q6 && k == 1) {                    // Q6_K_P1 = Q6_K_R1: todo byte de ql, qh e do plano vale
                p[208] = 0x12; p[209] = 0x1A;
            } else if (q6) {
                for (int i = 0; i < 256; i++) p[i] = (uint8_t) (int8_t) (rand() % 256 - 128);
                p[272] = 0x12; p[273] = 0x1A;
            } else {
                p[0] = 0x12; p[1] = 0x1A; p[2] = 0x34; p[3] = 0x18;
            }
        }
        int8_t tab[16];                            // tabela qualquer, |δ| ≤ 93, com a marca
        for (int i = 0; i < 16; i++) tab[i] = (int8_t) (rand() % 187 - 93);
        tab[15] = STRATA_TAB_MARCA;

        const float r = q6 ? strata_q6kd_vec_dot_q8_K(k, tab, x, y, cols) : strata_q4kd_vec_dot_q8_K(k, tab, x, y, cols);
        const float v = q6 ? strata_q6kd_vec_dot_q8_K_avx2(k, tab, x, y, cols) : strata_q4kd_vec_dot_q8_K_avx2(k, tab, x, y, cols);
        float * w = malloc(sizeof(float)*cols);
        if (q6) strata_q6kd_dequantize_row(k, tab, x, w, cols);
        else    strata_q4kd_dequantize_row(k, tab, x, w, cols);
        double s = 0;
        for (int b = 0; b < nb; b++) for (int i = 0; i < 256; i++) s += (double) w[b*256 + i]*y[b].d*y[b].qs[i];
        if (fabs(r - v)/fabs(r) > 1e-5 || fabs(r - s)/fabs(r) > 1e-5) {
            falhas++;
            printf("q6=%d k=%d: referência %g, AVX2 %g, desquantização %g\n", q6, k, r, v, s);
        }

        int8_t tp[16];                             // a tabela padrão tem que dar o tipo P
        strata_tab_padrao(k, tp);
        const float pd = q6 ? strata_q6kd_vec_dot_q8_K(k, tp, x, y, cols) : strata_q4kd_vec_dot_q8_K(k, tp, x, y, cols);
        const float pp = q6 ? strata_q6kp_vec_dot_q8_K(k, x, y, cols) : strata_q4kp_vec_dot_q8_K(k, x, y, cols);
        const float pv = q6 ? strata_q6kp_vec_dot_q8_K_avx2(k, x, y, cols) : strata_q4kp_vec_dot_q8_K_avx2(k, x, y, cols);
        if (fabs(pd - pp)/fabs(pp) > 1e-5 || fabs(pv - pp)/fabs(pp) > 1e-5) {
            falhas++;
            printf("tipo P q6=%d k=%d: D com a tabela padrão %g, P %g, P AVX2 %g\n", q6, k, pd, pp, pv);
        }
        if (q6 && k == 1) {
            const float rr = strata_q6kr_vec_dot_q8_K(1, x, y, cols);
            if (rr != pp) {
                falhas++;
                printf("Q6_K_P1 %g contra Q6_K_R1 %g\n", pp, rr);
            }
        }
        free(x); free(w);
    }
    printf(falhas ? "%d casos divergem\n" : "ok: D (escalar, AVX2, desquantização) e tabela padrão = P\n", falhas);
    return falhas != 0;
}
