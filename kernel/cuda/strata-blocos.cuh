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
