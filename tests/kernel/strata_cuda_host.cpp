// Confere, na CPU e sem GPU, a conta dos kernels CUDA do STRATA (kernel/cuda/*.cuh): desquantização e produto escalar
// contra ativações q8_1, comparados com a referência em C (kernel/strata_q4kr.c). Os cabeçalhos CUDA são compilados como
// C++ comum com a imitação de tests/kernel/cuda_shim/common.cuh; o lançamento <<<...>>> é tirado por sed antes.
//
//   sh tests/kernel/strata_cuda_host.sh            # compila e roda; sai com código 1 se algo divergir
//   ./strata_cuda_host [arquivo_de_blocos tipo]    # opcional: blocos reais (tipo 43..50; 47..50 com a tabela padrão)
#include "common.cuh"
#include "strata-dequant.cuh"
#include "strata-vecdot.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "strata_q4kr.h"
}

static uint32_t rs = 12345;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

static uint16_t f2h(float f) {   // só valores normais pequenos, o bastante para d e dmin
    int e; float m = std::frexp(f, &e);
    const int E = e + 14; const int M = (int) std::lround((2*m - 1) * 1024);
    return (uint16_t) ((E << 10) | (M & 0x3ff));
}

// tipos 47 a 50 (D) usam os blocos de 43 a 46 (P) e uma tabela de 16 bytes
static int base_p(int tipo) { return tipo >= 47 ? tipo - 4 : tipo; }
static int bloco(int tipo) { tipo = base_p(tipo); return tipo == 43 ? 176 : tipo == 44 ? 208 : tipo == 45 ? 242 : 274; }

static void aleatorio(int tipo, uint8_t * x, int nb) {
    const int bs = bloco(tipo);
    tipo = base_p(tipo);
    for (int i = 0; i < nb; i++) {
        uint8_t * b = x + (size_t) i*bs;
        for (int j = 0; j < bs; j++) b[j] = rnd();
        if (tipo == 45) {                         // o bloco Q6_K_R1: todo byte de ql, qh e do plano vale
            const uint16_t d = f2h(0.003f); memcpy(b + 208, &d, 2);
        } else if (tipo == 46) {
            for (int j = 0; j < 256; j++) b[j] = (uint8_t) (int8_t) ((int) (rnd() % 256) - 128);
            const uint16_t d = f2h(0.003f); memcpy(b + 272, &d, 2);
        } else {
            const uint16_t d = f2h(0.003f), m = f2h(0.002f); memcpy(b, &d, 2); memcpy(b + 2, &m, 2);
        }
    }
}

template <int K, bool Q6, bool D>
static void desq_cuda(const uint8_t * x, float * y, int nb, const strata_tab & tab) {
    for (int b = 0; b < nb; b++) {
        blockIdx.x = b;
        for (int t = 0; t < 64; t++) {
            threadIdx.x = t;
            if constexpr (D) {
                if constexpr (Q6) strata_dequantize_block_q6kd<K, float>(x, y, tab);
                else              strata_dequantize_block_q4kd<K, float>(x, y, tab);
            } else {
                if constexpr (Q6) strata_dequantize_block_q6kp<K, float>(x, y);
                else              strata_dequantize_block_q4kp<K, float>(x, y);
            }
        }
    }
}

template <int K, bool Q6, bool D>
static double dot_cuda(const uint8_t * x, const block_q8_1 * y, int nb, const strata_tab & tab) {
    double s = 0;
    for (int b = 0; b < nb; b++)
        for (int iqs = 0; iqs < 32; iqs += 2) {  // qi = 32, vdr = 2, como no mul_mat_vec_q
            if constexpr (D) s += Q6 ? vec_dot_strata_q6kd_q8_1<K>(x, y + 8*b, b, iqs, tab) : vec_dot_strata_q4kd_q8_1<K>(x, y + 8*b, b, iqs, tab);
            else             s += Q6 ? vec_dot_strata_q6kp_q8_1<K>(x, y + 8*b, b, iqs) : vec_dot_strata_q4kp_q8_1<K>(x, y + 8*b, b, iqs);
        }
    return s;
}

template <bool D>
static double tudo_cuda(int tipo, const uint8_t * x, float * y, const block_q8_1 * q, int nb, const strata_tab & tab) {
    switch (base_p(tipo)) {
        case 43: desq_cuda<1, false, D>(x, y, nb, tab); return dot_cuda<1, false, D>(x, q, nb, tab);
        case 44: desq_cuda<2, false, D>(x, y, nb, tab); return dot_cuda<2, false, D>(x, q, nb, tab);
        case 45: desq_cuda<1, true,  D>(x, y, nb, tab); return dot_cuda<1, true,  D>(x, q, nb, tab);
        default: desq_cuda<2, true,  D>(x, y, nb, tab); return dot_cuda<2, true,  D>(x, q, nb, tab);
    }
}

static int confere(int tipo, std::vector<uint8_t> & x, int nb, const int8_t * tab8) {
    const int n = nb*256, k = (base_p(tipo) == 43 || base_p(tipo) == 45) ? 1 : 2;
    const bool q6 = base_p(tipo) >= 45, d = tipo >= 47;
    std::vector<float> ref(n), cu(n);
    if (d) (q6 ? strata_q6kd_dequantize_row : strata_q4kd_dequantize_row)(k, tab8, x.data(), ref.data(), n);
    else   (q6 ? strata_q6kp_dequantize_row : strata_q4kp_dequantize_row)(k, x.data(), ref.data(), n);
    strata_tab tab;
    memcpy(&tab, tab8, sizeof(tab));

    // ativações q8_1 com alguns canais grandes, e o produto em float64 contra os pesos da referência
    std::vector<block_q8_1> q(n/32);
    double dref = 0, aref = 0;
    for (int b = 0; b < n/32; b++) {
        float a[32], amax = 0;
        for (int j = 0; j < 32; j++) { a[j] = ((int) (rnd() % 2001) - 1000) / 1000.0f * ((b*32 + j) % 97 == 0 ? 30.0f : 1.0f); amax = std::fmax(amax, std::fabs(a[j])); }
        const float d = amax / 127; int sum = 0;
        q[b].ds.x.b = f2h(d); q[b].ds.y.b = 0;
        const float dh = h2f(q[b].ds.x.b);
        for (int j = 0; j < 32; j++) { q[b].qs[j] = (int8_t) std::lround(a[j] / dh); sum += q[b].qs[j]; }
        for (int j = 0; j < 32; j++) { dref += (double) ref[b*32 + j] * dh * q[b].qs[j]; aref += std::fabs((double) ref[b*32 + j] * dh * q[b].qs[j]); }
    }
    const double dcu = d ? tudo_cuda<true>(tipo, x.data(), cu.data(), q.data(), nb, tab)
                         : tudo_cuda<false>(tipo, x.data(), cu.data(), q.data(), nb, tab);
    double emax = 0, wmax = 0;
    for (int i = 0; i < n; i++) { emax = std::fmax(emax, std::fabs(cu[i] - ref[i])); wmax = std::fmax(wmax, std::fabs(ref[i])); }
    const double erel_w = emax / wmax, erel_d = std::fabs(dcu - dref) / aref;
    const bool ok = erel_w < 1e-6 && erel_d < 1e-5;
    printf("tipo %d (%d blocos): desquantização erro máx %.2e do maior peso; produto escalar erro %.2e da soma dos |termos|  %s\n",
           tipo, nb, erel_w, erel_d, ok ? "OK" : "DIVERGE");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    int falhas = 0;
    if (argc > 2) {
        const int tipo = atoi(argv[2]);
        FILE * f = fopen(argv[1], "rb");
        if (!f) { perror(argv[1]); return 2; }
        fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> x(sz);
        if (fread(x.data(), 1, sz, f) != (size_t) sz) return 2;
        fclose(f);
        int8_t tab8[16];
        strata_tab_padrao(tipo == 43 || tipo == 45 || tipo == 47 || tipo == 49 ? 1 : 2, tab8);
        return confere(tipo, x, (int) (sz / bloco(tipo)), tab8);
    }
    for (int tipo = 43; tipo <= 50; tipo++) {
        const int nb = 512;
        std::vector<uint8_t> x((size_t) nb*bloco(tipo));
        aleatorio(tipo, x.data(), nb);
        int8_t tab8[16];                          // tabela qualquer, |δ| ≤ 93, com a marca
        for (int i = 0; i < 16; i++) tab8[i] = (int8_t) ((int) (rnd() % 187) - 93);
        tab8[15] = STRATA_TAB_MARCA;
        falhas += confere(tipo, x, nb, tab8);
    }
    return falhas ? 1 : 0;
}
