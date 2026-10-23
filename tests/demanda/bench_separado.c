// E2 da precisão sob demanda (docs/08): o layout separado (base do GGUF + um buffer de 32 bytes por bloco para cada nível)
// gera na velocidade dos blocos montados (tipos P)? Mede ns/peso do produto escalar matriz × vetor com os pesos fora do
// cache (matriz bem maior que o L3, como na geração de texto) e dentro dele, em 1 e em N threads (OpenMP).
// Antes, confere o resultado: S<k> igual bit a bit ao P<k> em AVX2 e perto da referência escalar do R<k> (erro relativo
// pequeno, a soma em float é a do tipo P).
//
//   cc -O2 -march=native -ffp-contract=off -fopenmp -I kernel tests/demanda/bench_separado.c kernel/strata_separado_avx2.c
//      kernel/strata_q4kr.c kernel/strata_q4kr_avx2.c -lm -o bench_separado
//   OMP_NUM_THREADS=1 ./bench_separado [linhas_frio] [colunas] [repetições]
#include "strata_separado.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t rs = 88172645463325252ULL;
static uint8_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint8_t)rs; }

// Matriz em três formas a partir dos mesmos bytes: base + planos separados, blocos R<k> (k = 1, 2) e blocos P<k>.
typedef struct {
    int q6, bbase;
    size_t nb;
    uint8_t *base, *pl[2], *plq[2], *r[3], *p[3];  // pl: arranjo do arquivo; plq: ordem qh (RAM)
} matriz;

static void monta(matriz *m, int q6, size_t nb) {
    m->q6 = q6, m->nb = nb, m->bbase = q6 ? STRATA_Q6K_BYTES : STRATA_Q4K_BYTES;
    m->base = malloc(nb * m->bbase);
    for (size_t i = 0; i < nb * m->bbase; i++) m->base[i] = rnd();
    for (size_t b = 0; b < nb; b++) {                 // d (e dmin) = fp16 pequeno e positivo, para não gerar NaN
        uint8_t *x = m->base + b * m->bbase;
        if (q6) x[208] = rnd(), x[209] = 0x20;
        else x[0] = rnd(), x[1] = 0x20, x[2] = rnd(), x[3] = 0x20;
    }
    for (int p = 0; p < 2; p++) {
        m->pl[p] = malloc(nb * STRATA_PLANO_BYTES);
        for (size_t i = 0; i < nb * STRATA_PLANO_BYTES; i++) m->pl[p][i] = rnd();
        m->plq[p] = malloc(nb * STRATA_PLANO_BYTES);
        strata_plano_para_qh(m->pl[p], m->plq[p], (int64_t)nb);
    }
    for (int k = 1; k <= 2; k++) {
        const int bb = m->bbase + STRATA_PLANO_BYTES * k;
        m->r[k] = malloc(nb * bb);
        for (size_t b = 0; b < nb; b++) {
            memcpy(m->r[k] + b * bb, m->base + b * m->bbase, m->bbase);
            for (int p = 0; p < k; p++) memcpy(m->r[k] + b * bb + m->bbase + 32 * p, m->pl[p] + 32 * b, 32);
        }
        const int pb = q6 ? strata_q6kp_block_bytes(k) : strata_q4kp_block_bytes(k);
        m->p[k] = malloc(nb * pb);
        if (q6) strata_q6kr_to_q6kp(k, m->r[k], m->p[k], (int64_t)nb * 256);
        else strata_q4kr_to_q4kp(k, m->r[k], m->p[k], (int64_t)nb * 256);
    }
}

static void solta(matriz *m) {
    free(m->base);
    for (int p = 0; p < 2; p++) free(m->pl[p]), free(m->plq[p]);
    for (int k = 1; k <= 2; k++) free(m->r[k]), free(m->p[k]);
}

// forma: 0 = S<k>, 1 = P<k>, 2 = R<k> (k = 0 em S é a base sozinha, o nível 0)
static float linha(const matriz *m, int forma, int k, size_t row, int cols, const strata_block_q8_K *y) {
    const size_t bpr = (size_t)cols / 256, b0 = row * bpr;
    if (forma == 0) {
        const uint8_t *pl[2] = {m->plq[0] + b0 * 32, m->plq[1] + b0 * 32};
        return m->q6 ? strata_q6ks_vec_dot_q8_K_avx2(k, m->base + b0 * m->bbase, pl, y, cols)
                     : strata_q4ks_vec_dot_q8_K_avx2(k, m->base + b0 * m->bbase, pl, y, cols);
    }
    if (forma == 1) {
        const int pb = m->q6 ? strata_q6kp_block_bytes(k) : strata_q4kp_block_bytes(k);
        return m->q6 ? strata_q6kp_vec_dot_q8_K_avx2(k, m->p[k] + b0 * pb, y, cols) : strata_q4kp_vec_dot_q8_K_avx2(k, m->p[k] + b0 * pb, y, cols);
    }
    const int rb = m->bbase + 32 * k;
    return m->q6 ? strata_q6kr_vec_dot_q8_K_avx2(k, m->r[k] + b0 * rb, y, cols) : strata_q4kr_vec_dot_q8_K_avx2(k, m->r[k] + b0 * rb, y, cols);
}

static double mede(const matriz *m, int forma, int k, int rows, int cols, const strata_block_q8_K *y, float *sink) {
    const double t0 = now();
    float s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int r = 0; r < rows; r++) s += linha(m, forma, k, r, cols, y);
    *sink += s;
    return now() - t0;
}

static int cmp_d(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double mediana(double *v, int n) { qsort(v, n, sizeof *v, cmp_d); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }

int main(int argc, char **argv) {
    const int rows_frio = argc > 1 ? atoi(argv[1]) : 65536, cols = argc > 2 ? atoi(argv[2]) : 4096, rows_quente = 1024;
    const int reps = argc > 3 ? atoi(argv[3]) : 15;
    float *act = malloc(sizeof(float) * cols);
    for (int i = 0; i < cols; i++) act[i] = (float)rnd() / 255.0f - 0.5f;
    strata_block_q8_K *y = malloc(sizeof(strata_block_q8_K) * cols / 256);
    strata_q8_K_quantize_row(act, y, cols);
    float sink = 0;

    // 1. conferência numa matriz pequena. S1/S2 contra P1/P2 (mesma conta, deve ser bit a bit) e contra a referência
    // escalar do R; S0 contra a referência escalar com k = 0 (o k-quant). O erro é medido contra o RMS das linhas, porque
    // com pesos aleatórios algumas linhas somam perto de zero.
    int falhas = 0;
    for (int q6 = 0; q6 < 2; q6++) {
        matriz m;
        monta(&m, q6, (size_t)256 * cols / 256);
        for (int k = 0; k <= 2; k++) {
            double se = 0, sr = 0, emax = 0;
            int difp = 0;
            for (int r = 0; r < 256; r++) {
                const float s = linha(&m, 0, k, r, cols, y);
                const size_t bpr = cols / 256, bb = (q6 ? STRATA_Q6K_BYTES : STRATA_Q4K_BYTES) + 32 * k;
                const uint8_t *src = k ? m.r[k] + r * bpr * bb : m.base + r * bpr * bb;
                const float ref = q6 ? strata_q6kr_vec_dot_q8_K(k, src, y, cols) : strata_q4kr_vec_dot_q8_K(k, src, y, cols);
                if (k) { const float p = linha(&m, 1, k, r, cols, y); if (memcmp(&s, &p, 4)) difp++; }
                const double e = fabs((double)s - ref);
                if (e > emax) emax = e;
                se += e * e, sr += (double)ref * ref;
            }
            const double rel = emax / sqrt(sr / 256);
            const int ok = rel < 1e-5 && difp == 0;
            falhas += !ok;
            printf("conferência %s S%d: %s  (erro máximo / RMS das linhas %.1e%s)\n", q6 ? "Q6_K" : "Q4_K", k, ok ? "ok" : "FALHOU", rel,
                   k ? (difp ? ", difere do AVX2 de referência" : q6 ? ", igual bit a bit ao Q6_K_P AVX2" : ", igual bit a bit ao Q4_K_P AVX2") : "");
        }
        solta(&m);
    }
    if (falhas) return 1;

    // 2. velocidade. As formas se revezam a cada repetição (a máquina é compartilhada e a velocidade muda com o tempo; o
    // rodízio espalha essa variação igualmente entre elas) e o resultado é a mediana das repetições.
    printf("\n%d thread(s); colunas %d; ns por peso (mediana de %d, formas em rodízio)\n", omp_get_max_threads(), cols, reps);
    printf("%-6s %-5s %-22s %9s %9s %9s\n", "grade", "nível", "forma", "frio", "quente", "P/S frio");
    enum { NC = 7 };
    for (int q6 = 0; q6 < 2; q6++) {
        matriz m;
        monta(&m, q6, (size_t)rows_frio * cols / 256);
        const double w_f = (double)rows_frio * cols, w_q = (double)rows_quente * cols;
        struct { int forma, k; const char *nome; } casos[NC] = {
            {0, 0, "base sozinha (S0)"}, {1, 1, "montado P1"}, {0, 1, "separado S1"}, {2, 1, "intercalado R1"},
            {1, 2, "montado P2"}, {0, 2, "separado S2"}, {2, 2, "intercalado R2"}};
        double *tf = malloc(sizeof(double) * NC * reps), *tq = malloc(sizeof(double) * NC * reps), mf[NC], mq[NC];
        for (int c = 0; c < NC; c++) mede(&m, casos[c].forma, casos[c].k, rows_quente, cols, y, &sink);  // aquece
        for (int rep = 0; rep < reps; rep++)
            for (int c = 0; c < NC; c++) {
                tf[c * reps + rep] = mede(&m, casos[c].forma, casos[c].k, rows_frio, cols, y, &sink);
                tq[c * reps + rep] = mede(&m, casos[c].forma, casos[c].k, rows_quente, cols, y, &sink);
            }
        for (int c = 0; c < NC; c++) mf[c] = mediana(tf + c * reps, reps), mq[c] = mediana(tq + c * reps, reps);
        for (int c = 0; c < NC; c++) {
            printf("%-6s %-5d %-22s %9.4f %9.4f", q6 ? "Q6_K" : "Q4_K", casos[c].k, casos[c].nome, 1e9 * mf[c] / w_f, 1e9 * mq[c] / w_q);
            if (casos[c].forma == 0 && casos[c].k) printf(" %9.3f", mf[c == 2 ? 1 : 4] / mf[c]);  // velocidade S / velocidade P
            printf("\n");
        }
        free(tf), free(tq);
        solta(&m);
    }

    // 3. custo de subir um nível: converter um plano do arranjo do arquivo para a ordem qh (uma vez, fora da geração).
    {
        const size_t nb = (size_t)rows_frio * cols / 256;
        uint8_t *a = malloc(nb * 32), *b = malloc(nb * 32);
        for (size_t i = 0; i < nb * 32; i++) a[i] = rnd();
        double t[5];
        for (int rep = 0; rep < 5; rep++) { const double t0 = now(); strata_plano_para_qh(a, b, (int64_t)nb); t[rep] = now() - t0; }
        sink += b[nb * 16];
        printf("\nconversão de um plano para a ordem qh: %.4f ns por peso, 1 thread (%.1f MB por plano)\n",
               1e9 * mediana(t, 5) / ((double)nb * 256), nb * 32 / 1e6);
        free(a), free(b);
    }
    // 4. a alternativa do plano se S<k> não chegar a 0,95 do P: montar os blocos P<k> em segundo plano ao subir de nível.
    // Mede a montagem a partir dos blocos R<k> (a partir da base + planos separados custaria um pouco mais).
    for (int q6 = 0; q6 < 2; q6++) {
        matriz m;
        monta(&m, q6, (size_t)rows_frio * cols / 256);
        const int64_t n = (int64_t)rows_frio * cols;
        for (int k = 1; k <= 2; k++) {
            double t[5];
            for (int rep = 0; rep < 5; rep++) {
                const double t0 = now();
                if (q6) strata_q6kr_to_q6kp(k, m.r[k], m.p[k], n);
                else strata_q4kr_to_q4kp(k, m.r[k], m.p[k], n);
                t[rep] = now() - t0;
            }
            printf("montagem P%d do %s: %.4f ns por peso, 1 thread\n", k, q6 ? "Q6_K" : "Q4_K", 1e9 * mediana(t, 5) / n);
        }
        solta(&m);
    }
    fprintf(stderr, "(%g)\n", sink);
    return 0;
}
