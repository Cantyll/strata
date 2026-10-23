// Produto escalar do layout separado (tipos S, strata_separado.h) com ativações Q8_K, em AVX2.
//
// Q4_K: como o Q4_K_P<k>, com u = 2^k·q4 + r montado no registrador a partir da base e dos planos (o Q4_K_P guarda u
// pronto na RAM): Σ t·q8 = 2·Σ u·q8 + (1 − 2^k)·Σ q8. Com k = 0 é o Q4_K comum.
// Q6_K: como o Q6_K_P<k>, com w montado no registrador.
// Os planos ficam na RAM na "ordem qh" (byte l, bit j = peso 32j + l), a do qh do Q5_K: o bit de cada grupo de 32 pesos sai
// com um deslocamento e um and, como no Q5_K/Q4_K_P1. O arquivo guarda os planos na ordem do tipo R; strata_plano_para_qh
// converte um plano ao subir de nível (só o plano novo; a base não muda).
// Compilar com -mavx2 -mfma -mf16c (ou -march=native) e -ffp-contract=off.
#include "strata_separado.h"
#include <immintrin.h>
#include <string.h>

#ifndef STRATA_PF
#define STRATA_PF 8
#endif
#define PF(x, bb) do { for (int l_ = 0; l_ < (bb); l_ += 64) _mm_prefetch((const char *)(x) + STRATA_PF*(bb) + l_, _MM_HINT_T0); } while (0)

static inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

static inline float hsum_f32(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
}

static inline __m256i scale16(__m256i s, int j) {
    return _mm256_shuffle_epi8(s, _mm256_set1_epi16((short)(((2 * j + 1) << 8) | (2 * j))));
}

// Escalas j = 2G e 2G + 1 do Q6_K, uma em cada lane de 128 bits (como q6k_s2 em strata_q4kr_avx2.c).
static inline __attribute__((always_inline)) __m256i q6k_s2(__m256i sA, __m256i sB, int G) {
    const int m = (2 * G) % 8;
    return _mm256_shuffle_epi8(G >= 4 ? sB : sA,
        _mm256_setr_epi8(2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                         2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                         2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3,
                         2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3));
}

// r (de 0 a 2^k − 1) dos pesos 32·j + l, l = 0..31, com os planos na ordem qh: o bit j de cada byte.
static inline __attribute__((always_inline)) __m256i bits_r(int k, __m256i h1, __m256i h2, int j) {
    const __m256i um = _mm256_set1_epi8(1);
    const __m256i b1 = _mm256_and_si256(_mm256_srli_epi16(h1, j), um);
    if (k == 1) return b1;
    return _mm256_or_si256(_mm256_add_epi8(b1, b1), _mm256_and_si256(_mm256_srli_epi16(h2, j), um));
}

void strata_plano_para_qh(const uint8_t *src, uint8_t *dst, int64_t nblocos) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    for (int64_t b = 0; b < nblocos; b++, src += STRATA_PLANO_BYTES, dst += STRATA_PLANO_BYTES) {
        __m256i acc = _mm256_setzero_si256();
        for (int j = 0; j < 8; j++) {
            uint32_t w;
            memcpy(&w, src + 4 * j, 4);
            const __m256i v = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)w), shuf), mask), mask);
            acc = _mm256_or_si256(acc, _mm256_and_si256(v, _mm256_set1_epi8((char)(1 << j))));
        }
        _mm256_storeu_si256((__m256i *)dst, acc);
    }
}

static inline __attribute__((always_inline)) float q4ks_dot(const int k, const uint8_t *x, const uint8_t *const *planos,
                                                            const strata_block_q8_K *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 1));
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const uint8_t *p1 = k > 0 ? planos[0] : 0, *p2 = k > 1 ? planos[1] : 0;
    float sum = 0;
    __m256 accf = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += STRATA_Q4K_BYTES) {
        const float d = _cvtsh_ss((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = _cvtsh_ss((uint16_t)(x[2] | (x[3] << 8)));
        const uint8_t *qs = x + 16;
        const int8_t *q8 = y[b].qs;
        PF(x, STRATA_Q4K_BYTES);
        __m256i h1 = _mm256_setzero_si256(), h2 = h1;
        if (k > 0) { h1 = _mm256_loadu_si256((const __m256i *)(p1 + 32 * b)); _mm_prefetch((const char *)(p1 + 32 * (b + STRATA_PF)), _MM_HINT_T0); }
        if (k > 1) { h2 = _mm256_loadu_si256((const __m256i *)(p2 + 32 * b)); _mm_prefetch((const char *)(p2 + 32 * (b + STRATA_PF)), _MM_HINT_T0); }
        uint32_t ut[4];
        memcpy(ut, x + 4, 12);
        ut[3] = ((ut[2] >> 4) & 0x0f0f0f0f) | (((ut[1] >> 6) & 0x03030303) << 4);
        const uint32_t ua = ut[1] & 0x3f3f3f3f;
        ut[1] = (ut[2] & 0x0f0f0f0f) | (((ut[0] >> 6) & 0x03030303) << 4);
        ut[2] = ua;
        ut[0] &= 0x3f3f3f3f;
        const __m128i sm = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)ut));
        const __m128i mm = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(ut + 2)));
        const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128((const __m128i *)y[b].bsums), _mm_loadu_si128((const __m128i *)(y[b].bsums + 8)));
        __m128i hh2 = _mm_hadd_epi32(_mm_madd_epi16(sm, bs), _mm_madd_epi16(mm, bs));
        hh2 = _mm_hadd_epi32(hh2, hh2);
        const int32_t sum_c = _mm_cvtsi128_si32(hh2), sum_m = _mm_extract_epi32(hh2, 1);
        const __m256i sc256 = _mm256_broadcastsi128_si256(sm);
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 4; h++) {
            const __m256i q4b = _mm256_loadu_si256((const __m256i *)(qs + 32 * h));
            for (int s = 0; s < 2; s++) {
                const int j = 2 * h + s;
                const __m256i q4 = s ? _mm256_and_si256(_mm256_srli_epi16(q4b, 4), m4) : _mm256_and_si256(q4b, m4);
                const __m256i u = k ? _mm256_or_si256(_mm256_slli_epi16(q4, k), bits_r(k, h1, h2, j)) : q4;
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * j));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, v8), scale16(sc256, j)));
            }
        }
        const float dd = y[b].d * d * inv;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(2 * dd), _mm256_cvtepi32_ps(acc), accf);
        sum += dd * (float)((1 - (1 << k)) * sum_c) - y[b].d * dmin * (float)sum_m;
    }
    return sum + hsum_f32(accf);
}

// Q6_K, como os tipos P: k = 0 e 1 com w + 2^(k+5) = 2^k·u + r sem sinal (até 7 bits) num maddubs; k = 2 com
// w = 4u + r − 128 em int8 e o truque do sinal (|w|·sinal(q8, w)), como o Q6_K_P2. Σ t·q8 = 2·Σ w·q8 + (1 − 2^k)·Σ q8.
static inline __attribute__((always_inline)) float q6ks_dot(const int k, const uint8_t *x, const uint8_t *const *planos,
                                                            const strata_block_q8_K *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 1));
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3), x80 = _mm256_set1_epi8((char)0x80);
    const uint8_t *p1 = k > 0 ? planos[0] : 0, *p2 = k > 1 ? planos[1] : 0;
    // termo das bsums: (1 − 2^k), mais −2^(k+5)·2 quando w vai deslocado (k = 0, 1)
    const int32_t off = (1 - (1 << k)) - (k < 2 ? 2 * (32 << k) : 0);
    float sum = 0;
    __m256 accf = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += STRATA_Q6K_BYTES) {
        const float d = _cvtsh_ss((uint16_t)(x[208] | (x[209] << 8)));
        const int8_t *q8 = y[b].qs;
        PF(x, STRATA_Q6K_BYTES);
        __m256i h1 = _mm256_setzero_si256(), h2 = h1;
        if (k > 0) { h1 = _mm256_loadu_si256((const __m256i *)(p1 + 32 * b)); _mm_prefetch((const char *)(p1 + 32 * (b + STRATA_PF)), _MM_HINT_T0); }
        if (k > 1) { h2 = _mm256_loadu_si256((const __m256i *)(p2 + 32 * b)); _mm_prefetch((const char *)(p2 + 32 * (b + STRATA_PF)), _MM_HINT_T0); }
        const __m256i sw = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 192)));
        const float isum = (float)((int64_t)off * hsum_i32(_mm256_madd_epi16(sw, _mm256_loadu_si256((const __m256i *)y[b].bsums))));
        const __m256i sA = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sw));
        const __m256i sB = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(sw, 1));
        __m256i acc = _mm256_setzero_si256();
        for (int hh = 0; hh < 2; hh++) {
            const __m256i l0 = _mm256_loadu_si256((const __m256i *)(x + 64 * hh));
            const __m256i l1 = _mm256_loadu_si256((const __m256i *)(x + 64 * hh + 32));
            const __m256i hb = _mm256_loadu_si256((const __m256i *)(x + 128 + 32 * hh));
            for (int g = 0; g < 4; g++) {
                const int G = 4 * hh + g;
                const __m256i lo = g < 2 ? _mm256_and_si256(g ? l1 : l0, m4) : _mm256_and_si256(_mm256_srli_epi16(g == 2 ? l0 : l1, 4), m4);
                const __m256i u = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * g), m2), 4));
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * G));
                const __m256i s = q6k_s2(sA, sB, G);
                if (k < 2) {
                    const __m256i w = k ? _mm256_or_si256(_mm256_add_epi8(u, u), bits_r(1, h1, h2, G)) : u;
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w, v8), s));
                } else {
                    const __m256i w = _mm256_xor_si256(_mm256_or_si256(_mm256_slli_epi16(u, 2), bits_r(2, h1, h2, G)), x80);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(v8, w)), s));
                }
            }
        }
        const float dd = y[b].d * d * inv;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(2 * dd), _mm256_cvtepi32_ps(acc), accf);
        sum += dd * isum;
    }
    return sum + hsum_f32(accf);
}

float strata_q4ks_vec_dot_q8_K_avx2(int k, const uint8_t *base, const uint8_t *const *planos, const strata_block_q8_K *y, int64_t n) {
    switch (k) {
    case 0: return q4ks_dot(0, base, planos, y, n);
    case 1: return q4ks_dot(1, base, planos, y, n);
    default: return q4ks_dot(2, base, planos, y, n);
    }
}

float strata_q6ks_vec_dot_q8_K_avx2(int k, const uint8_t *base, const uint8_t *const *planos, const strata_block_q8_K *y, int64_t n) {
    switch (k) {
    case 0: return q6ks_dot(0, base, planos, y, n);
    case 1: return q6ks_dot(1, base, planos, y, n);
    default: return q6ks_dot(2, base, planos, y, n);
    }
}
