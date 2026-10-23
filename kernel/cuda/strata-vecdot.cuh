#pragma once
// STRATA: produto escalar dos blocos montados contra ativações q8_1 (geração, MMVQ). Incluído no fim de vecdotq.cuh.
// Q4_K_P<k> segue o vec_dot_q5_K_q8_1 (mesmas posições de qs e das escalas); Q6_K_P2 lê os int8 direto e Q6_K_P1
// tira os 7 bits de ql, qh e do plano.

#include "strata-blocos.cuh"

// Q6_K_P1: w + 64 = 2u + r de 4 pesos seguidos (i0 múltiplo de 4), um por byte (strata_q6kp1_wb, de 4 em 4).
static __device__ __forceinline__ int strata_q6kp1_wb4(const block_strata_q6kp1 * x, const int i0) {
    const int h = i0 / 128, g = (i0 % 128) / 32, l = i0 % 32;
    const int ql = get_int_b2(x->ql, (64*h + 32*(g & 1) + l) / 4);
    const int qh = get_int_b2(x->qh, (32*h + l) / 4);
    const int u  = ((ql >> (4*(g >> 1))) & 0x0F0F0F0F) | (((qh >> (2*g)) & 0x03030303) << 4);
    const int nb = (x->r[i0 >> 3] >> (i0 & 4)) & 0xF;
    return (u << 1) | (nb & 1) | ((nb & 2) << 7) | ((nb & 4) << 14) | ((nb & 8) << 21);
}

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
    const int s = iqs/2;
    const block_q8_1 * bq8 = bq8_1 + s/2;
    const int * q8 = (const int *) bq8->qs + 4*(s%2);

    int sumi = 0, suma = 0;
    if constexpr (K == 1) {                       // w + 64 = 2u + r sem sinal: Σ (2w − 1)·a = 2·Σ (w + 64)·a − 129·Σ a
        const block_strata_q6kp1 * x = (const block_strata_q6kp1 *) vbq + kbx;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            sumi = ggml_cuda_dp4a(strata_q6kp1_wb4(x, 16*s + 4*t), q8[t], sumi);
            suma = ggml_cuda_dp4a(0x01010101, q8[t], suma);
        }
        const float d = __half2float(x->d) * STRATA_PASSO(1);
        return __low2float(bq8->ds) * d * x->sc[s] * (float) (2*sumi - 129*suma);
    } else {
        const block_strata_q6kp2 * x = (const block_strata_q6kp2 *) vbq + kbx;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            sumi = ggml_cuda_dp4a(get_int_b2(x->w, 4*s + t), q8[t], sumi);   // o bloco tem 274 bytes: só 2 bytes alinhados
            suma = ggml_cuda_dp4a(0x01010101, q8[t], suma);
        }
        const float d = __half2float(x->d) * STRATA_PASSO(K);
        return __low2float(bq8->ds) * d * x->sc[s] * (float) (2*sumi + STRATA_MEIO(K)*suma);
    }
}

// ---- Sub-células por tensor: Σ v·q8 = d·sc·(Σ c·q8 + Σ δ·q8/2^(k+4)) − dmin·mn·Σ q8, dois dp4a por grupo de 4 pesos.
// A classe de cada byte sai de dois __vcmpeq4 (c = menor, c = maior) e o δ de strata_tab_4.

template <int K>
static __device__ __forceinline__ float vec_dot_strata_q4kd_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const strata_tab & tab) {

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
    constexpr int RMASK = K == 1 ? 0x01010101 : 0x03030303;
    constexpr int K2    = (1 << K) * 0x01010101;
    float sumf_c = 0.0f;   // Σ d8·sc·Σ c·q8
    float sumf_e = 0.0f;   // Σ d8·sc·Σ δ·q8
    float sumf_m = 0.0f;   // Σ d8·mn·Σ q8
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        const float d8 = __low2float(bq8i->ds);
        const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);

        int dotc = 0, dote = 0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int u = ((ql[4*h] >> (4*i)) & 0x0F0F0F0F) | (((vh[h] >> (K*i)) << 4) & HMASK);
            const int c = (u >> K) & 0x0F0F0F0F;
            const int idx = ((u & RMASK) | K2) + (int) (__vcmpeq4(c, 0x0F0F0F0F) & K2) - (int) (__vcmpeq4(c, 0) & K2);
            dotc = ggml_cuda_dp4a(c, q8[4*h], dotc);
            dote = ggml_cuda_dp4a(strata_tab_4(tab, idx), q8[4*h], dote);
        }
        const int dot2 = ggml_cuda_dp4a(0x01010101, q8[0], ggml_cuda_dp4a(0x01010101, q8[4], 0));

        sumf_c += d8 * (dotc * sc[i]);
        sumf_e += d8 * (dote * sc[i]);
        sumf_m += d8 * (dot2 * m[i]);
    }

    const float2 dmf = __half22float2(dm);
    return dmf.x * (sumf_c + sumf_e * (1.0f / (float) (1 << (K + 4)))) - dmf.y*sumf_m;
}

template <int K>
static __device__ __forceinline__ float vec_dot_strata_q6kd_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const strata_tab & tab) {

    using bloco = std::conditional_t<K == 1, block_strata_q6kp1, block_strata_q6kp2>;
    const bloco * x = (const bloco *) vbq + kbx;
    const int s = iqs/2;
    const block_q8_1 * bq8 = bq8_1 + s/2;
    const int * q8 = (const int *) bq8->qs + 4*(s%2);

    constexpr int RMASK = K == 1 ? 0x01010101 : 0x03030303;
    constexpr int K2    = (1 << K) * 0x01010101;
    int sumc = 0, sume = 0, suma = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        int w, qp;                                                               // w: r nos k bits baixos; qp = q + 32
        if constexpr (K == 1) {
            w  = strata_q6kp1_wb4(x, 16*s + 4*t);                                // 2(q + 32) + r
            qp = (w >> 1) & 0x3F3F3F3F;
        } else {
            w  = get_int_b2(x->w, 4*s + t);
            qp = (int) ((((uint32_t) w ^ 0x80808080u) >> 2) & 0x3F3F3F3Fu);      // (w + 128) >> 2
        }
        const int idx = ((w & RMASK) | K2) + (int) (__vcmpeq4(qp, 0x3F3F3F3F) & K2) - (int) (__vcmpeq4(qp, 0) & K2);
        sumc = ggml_cuda_dp4a(qp, q8[t], sumc);
        sume = ggml_cuda_dp4a(strata_tab_4(tab, idx), q8[t], sume);
        suma = ggml_cuda_dp4a(0x01010101, q8[t], suma);
    }
    const float d = __half2float(x->d);
    return __low2float(bq8->ds) * d * x->sc[s] *
        ((float) (sumc - 32*suma) + (float) sume * (1.0f / (float) (1 << (K + 4))));
}

template <ggml_type type>
static __device__ __forceinline__ float vec_dot_strata_d_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const strata_tab & tab) {
    if constexpr (type == GGML_TYPE_STRATA_Q4_K_D1) return vec_dot_strata_q4kd_q8_1<1>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q4_K_D2) return vec_dot_strata_q4kd_q8_1<2>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q6_K_D1) return vec_dot_strata_q6kd_q8_1<1>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q6_K_D2) return vec_dot_strata_q6kd_q8_1<2>(vbq, bq8_1, kbx, iqs, tab);
    return 0.0f;
}
