#pragma once
// STRATA: blocos montados na carga (kernel/strata_q4kr.h no repositório do STRATA), como o ggml-cuda os enxerga.
//   Q4_K_P1: o mesmo arranjo do block_q5_K; u = 2·q4 + r (5 bits)
//   Q4_K_P2: d, dmin, escalas/mínimos do Q4_K, qs[128] (4 bits baixos de u), qh[64] (2 bits altos); u = 4·q4 + r
//   Q6_K_P1/P2: w = 2^k·q + r em int8 na ordem natural dos pesos, 16 escalas int8, d
// Valor: Q4_K_P<k> = d·sc·(2u + 1 − 2^k)/2^(k+1) − dmin·mn; Q6_K_P<k> = d·sc·(2w + 1 − 2^k)/2^(k+1).

#include "common.cuh"

struct block_strata_q4kp2 {
    half2   dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
    uint8_t qh[QK_K/4];
};
static_assert(sizeof(block_strata_q4kp2) == 208, "tamanho do bloco STRATA_Q4_K_P2");

struct block_strata_q6kp {
    int8_t w[QK_K];
    int8_t sc[QK_K/16];
    half   d;
};
static_assert(sizeof(block_strata_q6kp) == 274, "tamanho do bloco STRATA_Q6_K_P");
static_assert(sizeof(block_q5_K) == 176, "STRATA_Q4_K_P1 usa o arranjo do block_q5_K");

#define STRATA_PASSO(K)  (1.0f / (float) (1 << ((K) + 1)))      // 1/2^(k+1)
#define STRATA_MEIO(K)   ((float) (1 - (1 << (K))))              // 1 − 2^k

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P1> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_q5_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P2> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_strata_q4kp2);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P1> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_strata_q6kp);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P2> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_strata_q6kp);
};

// ---- Sub-células por tensor (Q4_K_D<k>, Q6_K_D<k>): mesmos blocos dos tipos P, mais uma tabela de 16 bytes do tensor
// (kernel/strata_q4kr.h). Valor: Q4_K_D<k> = d·sc·(c + δ/2^(k+4)) − dmin·mn, Q6_K_D<k> = d·sc·(q + δ/2^(k+4)), com
// δ = tab[r + 2^k·classe] (classe 0 = menor código, 2 = maior, 1 = os outros). A tabela vem do op_params do tensor de
// pesos (o carregador do llama.cpp a põe lá) e chega aos kernels por valor: no MMVQ pelo campo strata_tab dos argumentos
// de fusão; na desquantização pelo strata_cuda_tab_atual, que o ggml_cuda_mul_mat preenche antes de cada multiplicação.

#define STRATA_TAB_MARCA 0x53

struct strata_tab { uint32_t w[4]; };

static constexpr __host__ __device__ bool strata_tipo_d(ggml_type t) {
    return t == GGML_TYPE_STRATA_Q4_K_D1 || t == GGML_TYPE_STRATA_Q4_K_D2 ||
           t == GGML_TYPE_STRATA_Q6_K_D1 || t == GGML_TYPE_STRATA_Q6_K_D2;
}

// Tabela da multiplicação em curso (lado do host) e o intervalo de memória do tensor dono dela, para a desquantização
// conferir que a tabela é a do tensor que ela está lendo.
struct strata_tab_host { strata_tab tab; const char * ini; const char * fim; };
inline thread_local strata_tab_host strata_cuda_tab_atual = {};

static strata_tab strata_tab_de(const ggml_tensor * t) {
    strata_tab tab;
    memcpy(&tab, t->op_params, sizeof(tab));
    if ((int8_t) (tab.w[3] >> 24) != STRATA_TAB_MARCA) {
        GGML_ABORT("STRATA: o tensor %s (tipo %s) chegou sem a tabela de sub-células", t->name, ggml_type_name(t->type));
    }
    return tab;
}

static void strata_cuda_tab_definir(const ggml_tensor * src0) {
    if (!strata_tipo_d(src0->type)) {
        return;
    }
    strata_cuda_tab_atual.tab = strata_tab_de(src0);
    strata_cuda_tab_atual.ini = (const char *) src0->data;
    strata_cuda_tab_atual.fim = (const char *) src0->data + ggml_nbytes(src0);
}

static strata_tab strata_cuda_tab_para(const void * vx) {
    const char * p = (const char *) vx;
    GGML_ASSERT(p >= strata_cuda_tab_atual.ini && p < strata_cuda_tab_atual.fim && "STRATA: tabela de outro tensor");
    return strata_cuda_tab_atual.tab;
}

// 4 índices de 0 a 15 (um por byte de idx) → os 4 bytes correspondentes da tabela, empacotados como no dp4a.
static __device__ __forceinline__ int strata_tab_4(const strata_tab & t, const int idx) {
    const uint32_t i = (uint32_t) idx;
    const uint32_t sel = (i & 0x7) | ((i >> 4) & 0x70) | ((i >> 8) & 0x700) | ((i >> 12) & 0x7000);
    const uint32_t lo = __byte_perm(t.w[0], t.w[1], sel);
    const uint32_t hi = __byte_perm(t.w[2], t.w[3], sel);
    const uint32_t m = ((i >> 3) & 0x01010101) * 0xFF;
    return (int) ((hi & m) | (lo & ~m));
}

static __device__ __forceinline__ int strata_tab_1(const strata_tab & t, const int idx) {
    return (int) (int8_t) (t.w[idx >> 2] >> (8 * (idx & 3)));
}

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_D1> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P1> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_D2> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P2> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_D1> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P1> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_D2> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P2> {};
