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

// ---- Sub-células por tensor. Em unidades de 1/2^(k+4) do passo d·sc, o valor de um peso é o do tipo P mais o desvio da
// tabela: T = 2^(k+4)·c + δ = 16·u + 8·(1 − 2^k) + e (Q4_K, u = 2^k·c + r) ou 16·w + 8·(1 − 2^k) + e (Q6_K, w = 2^k·q + r),
// com e = δ − 8·(2r + 1 − 2^k) da tabela que o host prepara (strata_tab_d_mmvq). Então Σ T·q8 = 16·Σ u·q8 (o dp4a do tipo P)
// + Σ e·q8 + 8·(1 − 2^k)·Σ q8, e a consulta é a única conta a mais. O índice é r + 2^k·c', com c' = 0, 1 ou 3 (código menor,
// do meio ou maior): os dois bits de c' são o bit alto de u + (2^(k+4) − 2^k) (código ≥ 1) e de u + 2^k (código = 15), sem
// __vcmpeq4. As consultas são feitas de 8 em 8 pesos (dois grupos de 4 juntos em nibbles), com 1 __byte_perm por 4 pesos
// quando o índice cabe em 3 bits (k = 1) e 3 quando não cabe (k = 2), como o get_int_from_table_16 do ggml.
// LINHA (tabela com as três classes iguais, strata_tipo_mmvq): o índice é só r; com k = 1 o desvio é afim em r e vira um
// dp4a com o plano r, sem consulta.

template <int K>
static __device__ __forceinline__ int2 strata_d_consulta(const strata_tab & tab, const uint32_t i0, const uint32_t i1) {
    const int z = (int) (i0 | (i1 << 4));
    if constexpr (K == 1) {
        return strata_tab_8(tab, z);
    } else {
        return strata_tab_16(tab, 0, z);
    }
}

template <int K, bool LINHA = false>
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

    constexpr int      HMASK = K == 1 ? 0x10101010 : 0x30303030;
    constexpr uint32_t RMASK = ((1u << K) - 1) * 0x01010101u;
    constexpr uint32_t GE1   = ((1u << (K + 4)) - (1u << K)) * 0x01010101u;   // + u: bit k+4 ligado ⇔ código ≥ 1
    constexpr uint32_t EQ15  = (1u << K) * 0x01010101u;                        // + u: bit k+4 ligado ⇔ código = 15
    constexpr int      MEIO  = 8 * (1 - (1 << K));
    float sumf_d = 0.0f;   // Σ d8·sc·Σ T·q8
    float sumf_m = 0.0f;   // Σ d8·mn·Σ q8
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        const float d8 = __low2float(bq8i->ds);
        const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);

        uint32_t u[2], idx[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            u[h] = (uint32_t) (((ql[4*h] >> (4*i)) & 0x0F0F0F0F) | (((vh[h] >> (K*i)) << 4) & HMASK));
            if constexpr (LINHA) {
                idx[h] = u[h] & RMASK;
            } else {
                idx[h] = (u[h] & RMASK) | (((u[h] + GE1) >> 4) & EQ15) | (((u[h] + EQ15) >> 3) & (EQ15 << 1));
            }
        }
        const int dotu = ggml_cuda_dp4a((int) u[0], q8[0], ggml_cuda_dp4a((int) u[1], q8[4], 0));
        const int dot1 = ggml_cuda_dp4a(0x01010101, q8[0], ggml_cuda_dp4a(0x01010101, q8[4], 0));
        int soma;
        if constexpr (LINHA && K == 1) {
            // e(r) = e(0) + (e(1) − e(0))·r: um dp4a com o plano r, sem consulta
            const int dotr = ggml_cuda_dp4a((int) idx[0], q8[0], ggml_cuda_dp4a((int) idx[1], q8[4], 0));
            soma = 16*dotu + (int) tab.w[1]*dotr + (MEIO + (int) tab.w[0])*dot1;
        } else {
            const int2 e = strata_d_consulta<LINHA ? 1 : K>(tab, idx[0], idx[1]);   // LINHA: índice r < 4
            const int dote = ggml_cuda_dp4a(e.x, q8[0], ggml_cuda_dp4a(e.y, q8[4], 0));
            soma = 16*dotu + dote + MEIO*dot1;
        }

        sumf_d += d8 * (soma * sc[i]);
        sumf_m += d8 * (dot1 * m[i]);
    }

    const float2 dmf = __half22float2(dm);
    return dmf.x * sumf_d * (1.0f / (float) (1 << (K + 4))) - dmf.y*sumf_m;
}

template <int K, bool LINHA = false>
static __device__ __forceinline__ float vec_dot_strata_q6kd_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const strata_tab & tab) {

    using bloco = std::conditional_t<K == 1, block_strata_q6kp1, block_strata_q6kp2>;
    const bloco * x = (const bloco *) vbq + kbx;
    const int s = iqs/2;
    const block_q8_1 * bq8 = bq8_1 + s/2;
    const int * q8 = (const int *) bq8->qs + 4*(s%2);

    constexpr uint32_t RMASK = ((1u << K) - 1) * 0x01010101u;
    int sumw = 0, sume = 0, suma = 0;
#pragma unroll
    for (int t = 0; t < 4; t += 2) {
        int w[2];
        uint32_t idx[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if constexpr (K == 1) {
                // wb = w + 64 = 2(q + 32) + r, de 0 a 127: + 126 liga o bit 7 ⇔ q + 32 ≥ 1; + 2 ⇔ q + 32 = 63
                const uint32_t wb = (uint32_t) strata_q6kp1_wb4(x, 16*s + 4*(t + h));
                w[h] = (int) wb;
                idx[h] = LINHA ? wb & RMASK :
                    (wb & RMASK) | (((wb + 0x7E7E7E7Eu) >> 6) & 0x02020202u) | (((wb + 0x02020202u) >> 5) & 0x04040404u);
            } else {
                // w = 4q + r em int8; qp = q + 32 (6 bits): + 63 liga o bit 6 ⇔ qp ≥ 1; + 1 ⇔ qp = 63
                w[h] = get_int_b2(x->w, 4*s + t + h);   // o bloco tem 274 bytes: só 2 bytes alinhados
                if constexpr (LINHA) {
                    idx[h] = (uint32_t) w[h] & RMASK;
                } else {
                    const uint32_t qp = ((((uint32_t) w[h]) >> 2) ^ 0x20202020u) & 0x3F3F3F3Fu;
                    idx[h] = ((uint32_t) w[h] & RMASK) | (((qp + 0x3F3F3F3Fu) >> 4) & 0x04040404u) | (((qp + 0x01010101u) >> 3) & 0x08080808u);
                }
            }
        }
        sumw = ggml_cuda_dp4a(w[0], q8[t], ggml_cuda_dp4a(w[1], q8[t + 1], sumw));
        suma = ggml_cuda_dp4a(0x01010101, q8[t], ggml_cuda_dp4a(0x01010101, q8[t + 1], suma));
        if constexpr (LINHA && K == 1) {
            sume = ggml_cuda_dp4a((int) idx[0], q8[t], ggml_cuda_dp4a((int) idx[1], q8[t + 1], sume));   // Σ r·q8
        } else {
            const int2 e = strata_d_consulta<LINHA ? 1 : K>(tab, idx[0], idx[1]);
            sume = ggml_cuda_dp4a(e.x, q8[t], ggml_cuda_dp4a(e.y, q8[t + 1], sume));
        }
    }
    if constexpr (LINHA && K == 1) {   // Σ e·q8 = e(0)·Σ q8 + (e(1) − e(0))·Σ r·q8
        sume = (int) tab.w[0]*suma + (int) tab.w[1]*sume;
    }
    // K = 1: T = 2^5·q + δ = 16·wb + e − 8 − 1024 (wb = w + 64 sem sinal); K = 2: T = 16·w + e − 24 (w com sinal)
    constexpr int C = K == 1 ? -8 - 1024 : 8 * (1 - (1 << K));
    const float d = __half2float(x->d);
    return __low2float(bq8->ds) * d * x->sc[s] * (float) (16*sumw + sume + C*suma) * (1.0f / (float) (1 << (K + 4)));
}

// Livro de códigos aninhado: o vec_dot_iq4_xs_q8_1 do ggml com a tabela do tensor. iqs/4 = sub-bloco de 32 pesos; o int
// j (de 0 a 3) traz os pesos 4j..4j+3 nos nibbles baixos e 16+4j..16+4j+3 nos altos. No LIVRO1 a consulta é feita nas
// duas metades da tabela (pares e ímpares de T1) e o bit do plano escolhe o byte.
#define VDR_STRATA_LIVRO_Q8_1_MMVQ 4

template <int K>
static __device__ __forceinline__ float vec_dot_strata_livro_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const strata_tab & tab) {

    const block_iq4_xs * bq4;
    uint32_t plano = 0;
    if constexpr (K == 0) {
        bq4 = (const block_iq4_xs *) vbq + kbx;
    } else {
        const block_strata_livro1 * x = (const block_strata_livro1 *) vbq + kbx;
        bq4 = &x->b;
        plano = x->plano[iqs/4];
    }

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        int2 v = strata_tab_16(tab, 0, aux_q4);
        if constexpr (K == 1) {
            const int2 vi = strata_tab_16(tab, 4, aux_q4);
            const uint32_t m0 = strata_mascara_4(plano >> (4*j));
            const uint32_t m1 = strata_mascara_4(plano >> (16 + 4*j));
            v.x = (int) (((uint32_t) vi.x & m0) | ((uint32_t) v.x & ~m0));
            v.y = (int) (((uint32_t) vi.y & m1) | ((uint32_t) v.y & ~m1));
        }
        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

template <ggml_type type>
static __device__ __forceinline__ float vec_dot_strata_d_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const strata_tab & tab) {
    if constexpr (type == GGML_TYPE_STRATA_Q4_K_D1) return vec_dot_strata_q4kd_q8_1<1>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q4_K_D2) return vec_dot_strata_q4kd_q8_1<2>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q6_K_D1) return vec_dot_strata_q6kd_q8_1<1>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q6_K_D2) return vec_dot_strata_q6kd_q8_1<2>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q4_K_D1_LINHA) return vec_dot_strata_q4kd_q8_1<1, true>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q4_K_D2_LINHA) return vec_dot_strata_q4kd_q8_1<2, true>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q6_K_D1_LINHA) return vec_dot_strata_q6kd_q8_1<1, true>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_Q6_K_D2_LINHA) return vec_dot_strata_q6kd_q8_1<2, true>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_LIVRO0)  return vec_dot_strata_livro_q8_1<0>(vbq, bq8_1, kbx, iqs, tab);
    if constexpr (type == GGML_TYPE_STRATA_LIVRO1)  return vec_dot_strata_livro_q8_1<1>(vbq, bq8_1, kbx, iqs, tab);
    return 0.0f;
}
