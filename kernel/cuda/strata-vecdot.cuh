#pragma once
// STRATA: produto escalar dos blocos montados contra ativações q8_1 (geração, MMVQ). Incluído no fim de vecdotq.cuh.
// Q4_K_P<k> segue o vec_dot_q5_K_q8_1 (mesmas posições de qs e das escalas); Q6_K_P<k> lê os int8 direto.

#include "strata-blocos.cuh"

#define VDR_STRATA_Q4KP_Q8_1_MMVQ 2
#define VDR_STRATA_Q6KP_Q8_1_MMVQ 2

template <int K>
static __device__ __forceinline__ float vec_dot_strata_q4kp_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    // iqs = 0, 2, ..., 30: sub-blocos bq8_offset e bq8_offset + 1, 8 pesos de cada (4 em o, 4 em o + 16)
    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int o = 4 * ((iqs/2)%4);

    const uint8_t * qs;
    const uint8_t * scales;
    half2 dm;
    int vh[2];
    if constexpr (K == 1) {
        const block_q5_K * x = (const block_q5_K *) vbq + kbx;
        qs = x->qs; scales = x->scales; dm = x->dm;
        const int * qh = (const int *)(x->qh + o);
        vh[0] = qh[0] >> bq8_offset;
        vh[1] = qh[4] >> bq8_offset;
    } else {
        const block_strata_q4kp2 * x = (const block_strata_q4kp2 *) vbq + kbx;
        qs = x->qs; scales = x->scales; dm = x->dm;
        const int * qh = (const int *)(x->qh + 32*(bq8_offset/4) + o);
        vh[0] = qh[0] >> (2*(bq8_offset%4));
        vh[1] = qh[4] >> (2*(bq8_offset%4));
    }
    const int * ql = (const int *)(qs + 16 * bq8_offset + o);

    // escalas e mínimos de 6 bits dos sub-blocos bq8_offset e bq8_offset + 1, como no q5_K
    const uint16_t * sc16 = (const uint16_t *) scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;
    const uint32_t s0 = sc16[jm + 0];
    const uint32_t s2 = sc16[jm + 2];
    const uint32_t s4 = sc16[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

    constexpr int HMASK = K == 1 ? 0x10101010 : 0x30303030;
    float sumf_d = 0.0f;   // Σ d8·sc·Σ u·q8
    float sumf_s = 0.0f;   // Σ d8·sc·Σ q8   (meio-passo)
    float sumf_m = 0.0f;   // Σ d8·mn·Σ q8
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        const float d8 = __low2float(bq8i->ds);
        const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);

        const int v0 = ((ql[0] >> (4*i)) & 0x0F0F0F0F) | (((vh[0] >> (K*i)) << 4) & HMASK);
        const int v1 = ((ql[4] >> (4*i)) & 0x0F0F0F0F) | (((vh[1] >> (K*i)) << 4) & HMASK);

        const int dot1 = ggml_cuda_dp4a(v0, q8[0], ggml_cuda_dp4a(v1, q8[4], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, q8[0], ggml_cuda_dp4a(0x01010101, q8[4], 0));

        sumf_d += d8 * (dot1 * sc[i]);
        sumf_s += d8 * (dot2 * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }

    const float2 dmf = __half22float2(dm);
    return dmf.x * STRATA_PASSO(K) * (2.0f*sumf_d + STRATA_MEIO(K)*sumf_s) - dmf.y*sumf_m;
}

template <int K>
static __device__ __forceinline__ float vec_dot_strata_q6kp_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    // iqs = 0, 2, ..., 30: sub-bloco s = iqs/2 (pesos 16s ... 16s + 15), metade s%2 do bloco q8_1 s/2
    const block_strata_q6kp * x = (const block_strata_q6kp *) vbq + kbx;
    const int s = iqs/2;
    const block_q8_1 * bq8 = bq8_1 + s/2;
    const int * q8 = (const int *) bq8->qs + 4*(s%2);

    int sumi = 0, suma = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        sumi = ggml_cuda_dp4a(get_int_b2(x->w, 4*s + t), q8[t], sumi);   // o bloco tem 274 bytes: só 2 bytes alinhados
        suma = ggml_cuda_dp4a(0x01010101, q8[t], suma);
    }
    const float d = __half2float(x->d) * STRATA_PASSO(K);
    return __low2float(bq8->ds) * d * x->sc[s] * (float) (2*sumi + STRATA_MEIO(K)*suma);
}
