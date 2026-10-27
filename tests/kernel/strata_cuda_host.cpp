// Confere, na CPU e sem GPU, a conta dos kernels CUDA do STRATA (kernel/cuda/*.cuh): desquantização e produto escalar
// contra ativações q8_1, comparados com a referência em C (kernel/strata_q4kr.c). Os cabeçalhos CUDA são compilados como
// C++ comum com a imitação de tests/kernel/cuda_shim/common.cuh; o lançamento <<<...>>> é tirado por sed antes.
//
//   sh tests/kernel/strata_cuda_host.sh            # compila e roda; sai com código 1 se algo divergir
//   ./strata_cuda_host [arquivo_de_blocos tipo]    # opcional: blocos reais (tipo 43..50; 47..50 com a tabela padrão)
// Os tipos do livro (51 e 52) são conferidos com blocos e tabelas aleatórios, com a tabela arrumada pelo strata_tab_de
// do lado host (o mesmo caminho do llama.cpp: op_params do tensor → strata_tab).
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

template <int K, bool Q6, bool D, bool L>
static double dot_cuda(const uint8_t * x, const block_q8_1 * y, int nb, const strata_tab & tab) {   // tab: a do MMVQ
    double s = 0;
    for (int b = 0; b < nb; b++)
        for (int iqs = 0; iqs < 32; iqs += 2) {  // qi = 32, vdr = 2, como no mul_mat_vec_q
            if constexpr (D) s += Q6 ? vec_dot_strata_q6kd_q8_1<K, L>(x, y + 8*b, b, iqs, tab) : vec_dot_strata_q4kd_q8_1<K, L>(x, y + 8*b, b, iqs, tab);
            else             s += Q6 ? vec_dot_strata_q6kp_q8_1<K>(x, y + 8*b, b, iqs) : vec_dot_strata_q4kp_q8_1<K>(x, y + 8*b, b, iqs);
        }
    return s;
}

// tab: a tabela δ do tensor (desquantização); tm: a que o MMVQ recebe (strata_tab_mmvq); L: kernel LINHA
template <bool D, bool L>
static double tudo_cuda(int tipo, const uint8_t * x, float * y, const block_q8_1 * q, int nb, const strata_tab & tab, const strata_tab & tm) {
    switch (base_p(tipo)) {
        case 43: desq_cuda<1, false, D>(x, y, nb, tab); return dot_cuda<1, false, D, L>(x, q, nb, tm);
        case 44: desq_cuda<2, false, D>(x, y, nb, tab); return dot_cuda<2, false, D, L>(x, q, nb, tm);
        case 45: desq_cuda<1, true,  D>(x, y, nb, tab); return dot_cuda<1, true,  D, L>(x, q, nb, tm);
        default: desq_cuda<2, true,  D>(x, y, nb, tab); return dot_cuda<2, true,  D, L>(x, q, nb, tm);
    }
}

// linha: a tabela tem as três classes iguais, e o produto escalar passa pelo kernel LINHA que o host escolhe
static int confere(int tipo, std::vector<uint8_t> & x, int nb, const int8_t * tab8, bool linha = false) {
    const int n = nb*256, k = (base_p(tipo) == 43 || base_p(tipo) == 45) ? 1 : 2;
    const bool q6 = base_p(tipo) >= 45, d = tipo >= 47;
    std::vector<float> ref(n), cu(n);
    if (d) (q6 ? strata_q6kd_dequantize_row : strata_q4kd_dequantize_row)(k, tab8, x.data(), ref.data(), n);
    else   (q6 ? strata_q6kp_dequantize_row : strata_q4kp_dequantize_row)(k, x.data(), ref.data(), n);
    strata_tab tab = {};
    memcpy(tab.w, tab8, 16);
    ggml_tensor t = {};
    t.type = (ggml_type) tipo;
    memcpy(t.op_params, tab8, 16);
    const ggml_type tk = d ? strata_tipo_mmvq(&t, nullptr) : t.type;
    if (strata_tipo_linha(tk) != linha) {
        printf("tipo %d: o host escolheu o kernel %d com a tabela %s\n", tipo, (int) tk, linha ? "linha" : "por classe");
        return 1;
    }
    const strata_tab tm = d ? strata_tab_mmvq(&t, tk) : tab;

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
    const double dcu = !d ? tudo_cuda<false, false>(tipo, x.data(), cu.data(), q.data(), nb, tab, tm)
                     : linha ? tudo_cuda<true, true>(tipo, x.data(), cu.data(), q.data(), nb, tab, tm)
                             : tudo_cuda<true, false>(tipo, x.data(), cu.data(), q.data(), nb, tab, tm);
    double emax = 0, wmax = 0;
    for (int i = 0; i < n; i++) { emax = std::fmax(emax, std::fabs(cu[i] - ref[i])); wmax = std::fmax(wmax, std::fabs(ref[i])); }
    const double erel_w = emax / wmax, erel_d = std::fabs(dcu - dref) / aref;
    const bool ok = erel_w < 1e-6 && erel_d < 1e-5;
    printf("tipo %d%s (%d blocos): desquantização erro máx %.2e do maior peso; produto escalar erro %.2e da soma dos |termos|  %s\n",
           tipo, linha ? " linha" : "", nb, erel_w, erel_d, ok ? "OK" : "DIVERGE");
    return ok ? 0 : 1;
}

// ativações q8_1 aleatórias (alguns canais grandes); dref = Σ peso·ativação em float64 com os pesos de ref
static std::vector<block_q8_1> ativacoes(const std::vector<float> & ref, double & dref, double & aref) {
    const int n = (int) ref.size();
    std::vector<block_q8_1> q(n/32);
    dref = aref = 0;
    for (int b = 0; b < n/32; b++) {
        float a[32], amax = 0;
        for (int j = 0; j < 32; j++) { a[j] = ((int) (rnd() % 2001) - 1000) / 1000.0f * ((b*32 + j) % 97 == 0 ? 30.0f : 1.0f); amax = std::fmax(amax, std::fabs(a[j])); }
        q[b].ds.x.b = f2h(amax / 127); q[b].ds.y.b = 0;
        const float dh = h2f(q[b].ds.x.b);
        for (int j = 0; j < 32; j++) q[b].qs[j] = (int8_t) std::lround(a[j] / dh);
        for (int j = 0; j < 32; j++) { dref += (double) ref[b*32 + j] * dh * q[b].qs[j]; aref += std::fabs((double) ref[b*32 + j] * dh * q[b].qs[j]); }
    }
    return q;
}

template <int K>
static int confere_livro(int nb) {
    const int bs = K == 0 ? 136 : 168, n = nb*256, nt = K == 0 ? 16 : 32;
    std::vector<uint8_t> x((size_t) nb*bs);
    for (auto & v : x) v = rnd();
    for (int b = 0; b < nb; b++) { const uint16_t d = f2h(0.003f); memcpy(&x[(size_t) b*bs], &d, 2); }
    ggml_tensor t = {};
    t.type = K == 0 ? GGML_TYPE_STRATA_LIVRO0 : GGML_TYPE_STRATA_LIVRO1;
    int8_t * op = (int8_t *) t.op_params;
    for (int i = 0; i < nt; i++) op[i] = (int8_t) ((int) (rnd() % 255) - 127);
    op[STRATA_LIVRO_MARCA_POS] = STRATA_LIVRO_MARCA;
    const strata_tab tab = strata_tab_de(&t);

    std::vector<float> ref(n), cu(n);
    strata_livro_dequantize_row(K, op, x.data(), ref.data(), n);
    double dref, aref;
    const std::vector<block_q8_1> q = ativacoes(ref, dref, aref);
    for (int b = 0; b < nb; b++) {
        blockIdx.x = b;
        for (int th = 0; th < 32; th++) { threadIdx.x = th; strata_dequantize_block_livro<K, float>(x.data(), cu.data(), tab); }
    }
    double dcu = 0;
    for (int b = 0; b < nb; b++)
        for (int iqs = 0; iqs < QI4_XS; iqs += VDR_STRATA_LIVRO_Q8_1_MMVQ)
            dcu += vec_dot_strata_livro_q8_1<K>(x.data(), q.data() + 8*b, b, iqs, tab);
    double emax = 0, wmax = 0;
    for (int i = 0; i < n; i++) { emax = std::fmax(emax, std::fabs(cu[i] - ref[i])); wmax = std::fmax(wmax, std::fabs(ref[i])); }
    const double erel_w = emax / wmax, erel_d = std::fabs(dcu - dref) / aref;
    const bool ok = erel_w < 1e-6 && erel_d < 1e-5;
    printf("tipo %d (livro %d, %d blocos): desquantização erro máx %.2e do maior peso; produto escalar erro %.2e da soma dos |termos|  %s\n",
           51 + K, K, nb, erel_w, erel_d, ok ? "OK" : "DIVERGE");
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
        if (tipo >= 47) {                         // a mesma tabela com as três classes iguais à do meio: kernel LINHA
            const int k = tipo == 47 || tipo == 49 ? 1 : 2;
            for (int r = 0; r < (1 << k); r++) tab8[r] = tab8[r + (2 << k)] = tab8[r + (1 << k)];
            falhas += confere(tipo, x, nb, tab8, true);
        }
    }
    for (int r = 0; r < 4; r++) falhas += confere_livro<0>(256) + confere_livro<1>(256);
    return falhas ? 1 : 0;
}
