#pragma once
// STRATA: desquantização dos blocos montados (leitura do prompt pelo cuBLAS e conversões). Incluído em convert.cu
// depois de dequantize.cuh. 64 threads por bloco de 256 pesos, 4 pesos por thread, como o Q5_K e o Q6_K.

#include "strata-blocos.cuh"

template<int K, typename dst_t>
static __global__ void strata_dequantize_block_q4kp(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t ib  = blockIdx.x;
    const int     tid = threadIdx.x;
    const int     il  = tid/16;      // pedaço de 64 pesos, 0...3
    const int     ir  = tid%16;      // 0...15
    const int     is  = 2*il;        // sub-blocos is e is + 1

    dst_t * y = yy + ib*QK_K + 64*il + 2*ir;

    const uint8_t * scales;
    const uint8_t * ql;
    float dall, dmin;
    int h0[2], h1[2];                // bits altos de u já na posição (<< 4)
    if constexpr (K == 1) {
        const block_q5_K * x = (const block_q5_K *) vx + ib;
        dall = __low2half(x->dm); dmin = __high2half(x->dm);
        scales = x->scales;
        ql = x->qs + 32*il + 2*ir;
        const uint8_t * qh = x->qh + 2*ir;
        for (int t = 0; t < 2; ++t) {
            h0[t] = ((qh[t] >> (is + 0)) & 1) << 4;
            h1[t] = ((qh[t] >> (is + 1)) & 1) << 4;
        }
    } else {
        const block_strata_q4kp2 * x = (const block_strata_q4kp2 *) vx + ib;
        dall = __low2half(x->dm); dmin = __high2half(x->dm);
        scales = x->scales;
        ql = x->qs + 32*il + 2*ir;
        const uint8_t * qh = x->qh + 32*(is/4) + 2*ir;   // is e is + 1 dividem o mesmo byte
        for (int t = 0; t < 2; ++t) {
            h0[t] = ((qh[t] >> (2*((is + 0)%4))) & 3) << 4;
            h1[t] = ((qh[t] >> (2*((is + 1)%4))) & 3) << 4;
        }
    }

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales, sc, m);
    const float d1 = dall*sc*STRATA_PASSO(K); const float c1 = dall*sc*STRATA_PASSO(K)*STRATA_MEIO(K) - dmin*m;
    get_scale_min_k4(is + 1, scales, sc, m);
    const float d2 = dall*sc*STRATA_PASSO(K); const float c2 = dall*sc*STRATA_PASSO(K)*STRATA_MEIO(K) - dmin*m;

    for (int t = 0; t < 2; ++t) {
        const int u0 = (ql[t] & 0xF) | h0[t];
        const int u1 = (ql[t] >>  4) | h1[t];
        y[t +  0] = ggml_cuda_cast<dst_t>(d1*(2*u0) + c1);   // d·sc·(2u + 1 − 2^k)/2^(k+1) − dmin·mn
        y[t + 32] = ggml_cuda_cast<dst_t>(d2*(2*u1) + c2);
    }
}

template<int K, typename dst_t>
static __global__ void strata_dequantize_block_q6kp(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t ib  = blockIdx.x;
    const int     tid = threadIdx.x;
    dst_t * y = yy + ib*QK_K;
    if constexpr (K == 1) {
        const block_strata_q6kp1 * x = (const block_strata_q6kp1 *) vx + ib;
        const float d = __half2float(x->d)*STRATA_PASSO(1);
        for (int t = 0; t < 4; ++t) {
            const int e = tid + 64*t;
            y[e] = ggml_cuda_cast<dst_t>(d*x->sc[e/16]*(float) (2*strata_q6kp1_wb(x, e) - 129));   // 2w − 1, w = wb − 64
        }
    } else {
        const block_strata_q6kp2 * x = (const block_strata_q6kp2 *) vx + ib;
        const float d = __half2float(x->d)*STRATA_PASSO(K);
        for (int t = 0; t < 4; ++t) {
            const int e = tid + 64*t;
            y[e] = ggml_cuda_cast<dst_t>(d*x->sc[e/16]*(float) (2*x->w[e] + 1 - (1 << K)));
        }
    }
}

template<int K, typename dst_t>
static void strata_dequantize_row_q4kp_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    strata_dequantize_block_q4kp<K><<<k / QK_K, 64, 0, stream>>>(vx, y);
}

template<int K, typename dst_t>
static void strata_dequantize_row_q6kp_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    strata_dequantize_block_q6kp<K><<<k / QK_K, 64, 0, stream>>>(vx, y);
}

// Sub-células por tensor: o mesmo arranjo de threads dos tipos P; o valor sai de c e δ = tab[r + 2^k·classe].
static __device__ __forceinline__ int strata_classe(const int c, const int cmin, const int cmax) {
    return c == cmin ? 0 : (c == cmax ? 2 : 1);
}

template<int K, typename dst_t>
static __global__ void strata_dequantize_block_q4kd(const void * __restrict__ vx, dst_t * __restrict__ yy, const strata_tab tab) {
    const int64_t ib  = blockIdx.x;
    const int     tid = threadIdx.x;
    const int     il  = tid/16;
    const int     ir  = tid%16;
    const int     is  = 2*il;

    dst_t * y = yy + ib*QK_K + 64*il + 2*ir;

    const uint8_t * scales;
    const uint8_t * ql;
    float dall, dmin;
    int h0[2], h1[2];
    if constexpr (K == 1) {
        const block_q5_K * x = (const block_q5_K *) vx + ib;
        dall = __low2half(x->dm); dmin = __high2half(x->dm);
        scales = x->scales;
        ql = x->qs + 32*il + 2*ir;
        const uint8_t * qh = x->qh + 2*ir;
        for (int t = 0; t < 2; ++t) {
            h0[t] = ((qh[t] >> (is + 0)) & 1) << 4;
            h1[t] = ((qh[t] >> (is + 1)) & 1) << 4;
        }
    } else {
        const block_strata_q4kp2 * x = (const block_strata_q4kp2 *) vx + ib;
        dall = __low2half(x->dm); dmin = __high2half(x->dm);
        scales = x->scales;
        ql = x->qs + 32*il + 2*ir;
        const uint8_t * qh = x->qh + 32*(is/4) + 2*ir;
        for (int t = 0; t < 2; ++t) {
            h0[t] = ((qh[t] >> (2*((is + 0)%4))) & 3) << 4;
            h1[t] = ((qh[t] >> (2*((is + 1)%4))) & 3) << 4;
        }
    }

    constexpr float inv = 1.0f / (float) (1 << (K + 4));
    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales, sc, m);
    const float d1 = dall*sc; const float m1 = dmin*m;
    get_scale_min_k4(is + 1, scales, sc, m);
    const float d2 = dall*sc; const float m2 = dmin*m;

    for (int t = 0; t < 2; ++t) {
        const int u0 = (ql[t] & 0xF) | h0[t];
        const int u1 = (ql[t] >>  4) | h1[t];
        const int c0 = u0 >> K, c1 = u1 >> K;
        const int t0 = (c0 << (K + 4)) + strata_tab_1(tab, (u0 & ((1 << K) - 1)) + (strata_classe(c0, 0, 15) << K));
        const int t1 = (c1 << (K + 4)) + strata_tab_1(tab, (u1 & ((1 << K) - 1)) + (strata_classe(c1, 0, 15) << K));
        y[t +  0] = ggml_cuda_cast<dst_t>(d1*((float) t0*inv) - m1);   // d·sc·(c + δ/2^(k+4)) − dmin·mn
        y[t + 32] = ggml_cuda_cast<dst_t>(d2*((float) t1*inv) - m2);
    }
}

template<int K, typename dst_t>
static __global__ void strata_dequantize_block_q6kd(const void * __restrict__ vx, dst_t * __restrict__ yy, const strata_tab tab) {
    const int64_t ib  = blockIdx.x;
    const int     tid = threadIdx.x;
    constexpr float inv = 1.0f / (float) (1 << (K + 4));
    dst_t * y = yy + ib*QK_K;
    using bloco = std::conditional_t<K == 1, block_strata_q6kp1, block_strata_q6kp2>;
    const bloco * x = (const bloco *) vx + ib;
    const float d = __half2float(x->d);
    for (int t = 0; t < 4; ++t) {
        const int e = tid + 64*t;
        int w;
        if constexpr (K == 1) w = strata_q6kp1_wb(x, e) - 64; else w = x->w[e];
        const int q = w >> K;
        const int v = q * (1 << (K + 4)) + strata_tab_1(tab, (w & ((1 << K) - 1)) + (strata_classe(q, -32, 31) << K));
        y[e] = ggml_cuda_cast<dst_t>(d*x->sc[e/16]*((float) v*inv));
    }
}

template<int K, typename dst_t>
static void strata_dequantize_row_q4kd_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    strata_dequantize_block_q4kd<K><<<k / QK_K, 64, 0, stream>>>(vx, y, strata_cuda_tab_para(vx));
}

template<int K, typename dst_t>
static void strata_dequantize_row_q6kd_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    strata_dequantize_block_q6kd<K><<<k / QK_K, 64, 0, stream>>>(vx, y, strata_cuda_tab_para(vx));
}

// Livro de códigos aninhado: o dequantize_iq4_xs do ggml (32 threads por bloco, 8 pesos cada) com a tabela do tensor.
template<int K, typename dst_t>
static __global__ void strata_dequantize_block_livro(const void * __restrict__ vx, dst_t * __restrict__ yy, const strata_tab tab) {
    const int64_t ibs = blockIdx.x;
    const int     tid = threadIdx.x;
    const int     il  = tid/8;   // 0...3
    const int     ib  = tid%8;   // 0...7
    const block_iq4_xs * x;
    uint32_t plano = 0;
    if constexpr (K == 0) {
        x = (const block_iq4_xs *) vx + ibs;
    } else {
        const block_strata_livro1 * b = (const block_strata_livro1 *) vx + ibs;
        x = &b->b;
        plano = b->plano[ib];
    }
    dst_t * y = yy + ibs*QK_K + 32*ib + 4*il;
    const uint8_t * q4 = x->qs + 16*ib + 4*il;
    const float d = __half2float(x->d) * ((((x->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x->scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        const int a = q4[j] & 0xf, b = q4[j] >> 4;
        const int fa = K == 1 ? (plano >> (4*il + j)) & 1 : 0;
        const int fb = K == 1 ? (plano >> (16 + 4*il + j)) & 1 : 0;
        y[j +  0] = ggml_cuda_cast<dst_t>(d * strata_tab_1(tab, a + 16*fa));
        y[j + 16] = ggml_cuda_cast<dst_t>(d * strata_tab_1(tab, b + 16*fb));
    }
}

template<int K, typename dst_t>
static void strata_dequantize_row_livro_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    strata_dequantize_block_livro<K><<<k / QK_K, 32, 0, stream>>>(vx, y, strata_cuda_tab_para(vx));
}
