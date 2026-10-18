// Produto escalar dos tipos Q4_K_R<k> e Q6_K_R<k> com ativações Q8_K, em AVX2. Dá o mesmo float, bit a bit, que a
// referência escalar (strata_q4kr.c): as somas inteiras por bloco são as mesmas e a conta final em float também.
//
// A soma de cada sub-bloco se separa em três partes, todas com produtos de 8 bits que cabem no _mm256_maddubs_epi16:
//   Σ t·q8 = 2^(k+1)·Σ q·q8 + 2·Σ r·q8 + (1 − 2^k)·Σ q8
// q·q8 é o produto escalar do Q4_K (ou Q6_K) de sempre, r·q8 é o dos bits dos níveis (r ≤ 7) e Σ q8 já vem pronto
// nas bsums do Q8_K. No Q6_K, q = u − 32 com u de 0 a 63, e o −32 entra no termo das bsums.
// Compilar com -mavx2 -mfma (ou -march=native) e -ffp-contract=off.
#include "strata_q4kr.h"
#include <immintrin.h>
#include <string.h>

static float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

// Soma dos 8 int32 de v.
static inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// r (de 0 a 2^k − 1) dos 32 pesos que começam no peso 32·g do bloco: nível 1 é o bit mais significativo.
// Cada plano entra como r = 2r + bit, com o bit vindo do cmpeq (−1 ou 0): r + r − (−1).
static inline __attribute__((always_inline)) __m256i refine32(int k, const uint8_t *planes, int g) {
    const __m256i shuf = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                          2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i r = _mm256_setzero_si256();
    for (int p = 0; p < k; p++) {
        uint32_t w;
        memcpy(&w, planes + 32 * p + 4 * g, 4);
        const __m256i v = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)w), shuf), mask);
        r = _mm256_sub_epi8(_mm256_add_epi8(r, r), _mm256_cmpeq_epi8(v, mask));
    }
    return r;
}

// A escala j (int16 na posição j de s) repetida nas 16 posições.
static inline __m256i scale16(__m256i s, int j) {
    return _mm256_shuffle_epi8(s, _mm256_set1_epi16((short)(((2 * j + 1) << 8) | (2 * j))));
}

// Com u = t + 2^k − 1 = 2^(k+1)·q4 + 2r (de 0 a 2^(k+2)·8 − 2), Σ sc·Σ t·q8 = Σ sc·Σ u·q8 + (1 − 2^k)·Σ sc·bsum.
// Até k = 2, u cabe em 7 bits e vai num maddubs só; com k = 3, q4 e r vão em dois maddubs.
static inline __attribute__((always_inline)) float q4kr_dot(const int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q4kr_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    const __m256i m4 = _mm256_set1_epi8(0xF);
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        const uint8_t *scales = x + 4, *qs = x + 16, *planes = x + STRATA_Q4K_BYTES;
        const int8_t *q8 = y[b].qs;
        // escalas e mínimos de 6 bits desembrulhados como no ggml: bytes 0..7 = escalas, 8..15 = mínimos
        uint32_t ut[4];
        memcpy(ut, scales, 12);
        ut[3] = ((ut[2] >> 4) & 0x0f0f0f0f) | (((ut[1] >> 6) & 0x03030303) << 4);
        const uint32_t ua = ut[1] & 0x3f3f3f3f;
        ut[1] = (ut[2] & 0x0f0f0f0f) | (((ut[0] >> 6) & 0x03030303) << 4);
        ut[2] = ua;
        ut[0] &= 0x3f3f3f3f;
        const __m256i sc256 = _mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)ut)));
        // Σ sc_j·bsum_j e Σ mn_j·bsum_j de uma vez (bsum de 32 pesos = soma de duas bsums de 16)
        const __m128i sm = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)ut));
        const __m128i mm = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(ut + 2)));
        const __m128i b0 = _mm_loadu_si128((const __m128i *)y[b].bsums), b1 = _mm_loadu_si128((const __m128i *)(y[b].bsums + 8));
        const __m128i bs = _mm_hadd_epi16(b0, b1);
        const __m128i pc = _mm_madd_epi16(sm, bs), pm = _mm_madd_epi16(mm, bs);
        __m128i h2 = _mm_hadd_epi32(pc, pm);
        h2 = _mm_hadd_epi32(h2, h2);
        const int32_t sum_c = _mm_cvtsi128_si32(h2), sum_m = _mm_extract_epi32(h2, 1);
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 4; h++) {                         // 64 pesos: sub-blocos 2h (nibble baixo) e 2h+1 (alto)
            const __m256i q4b = _mm256_loadu_si256((const __m256i *)(qs + 32 * h));
            for (int s = 0; s < 2; s++) {
                const int j = 2 * h + s;
                const __m256i q4 = s ? _mm256_and_si256(_mm256_srli_epi16(q4b, 4), m4) : _mm256_and_si256(q4b, m4);
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * j));
                const __m256i r = refine32(k, planes, j);
                if (k <= 2) {
                    const __m256i u = _mm256_add_epi8(_mm256_slli_epi16(q4, k + 1), _mm256_add_epi8(r, r));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, v8), scale16(sc256, j)));
                } else {
                    const __m256i s16 = scale16(sc256, j);
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(q4, v8), _mm256_slli_epi16(s16, k + 1)));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(r, v8), _mm256_slli_epi16(s16, 1)));
                }
            }
        }
        const int32_t sum_t = hsum_i32(acc) + (1 - (1 << k)) * sum_c;
        sum += y[b].d * (d * inv * (float)sum_t - dmin * (float)sum_m);
    }
    return sum;
}

// Q6_K: Σ t·q8 = 2^(k+1)·Σ u·q8 + 2·Σ r·q8 + off·Σ q8, com u = q + 32. Cada metade de 128 pesos soma em 32 bits
// (o pior caso de uma metade cabe) e o bloco em 64, como na referência.
static inline __attribute__((always_inline)) float q6kr_dot(const int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q6kr_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3);
    const int32_t off = 1 - (1 << k) - 32 * (1 << (k + 1));  // termo das bsums: (1 − 2^k) − 32·2^(k+1)
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[208] | (x[209] << 8)));
        const int8_t *sc = (const int8_t *)(x + 192);
        const uint8_t *planes = x + STRATA_Q6K_BYTES;
        const int8_t *q8 = y[b].qs;
        int64_t isum = 0;
        for (int j = 0; j < 16; j++) isum += (int64_t)sc[j] * (off * y[b].bsums[j]);
        for (int hh = 0; hh < 2; hh++) {                      // metades de 128 pesos
            const uint8_t *ql = x + 64 * hh, *qh = x + 128 + 32 * hh;
            const __m256i l0 = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i hb = _mm256_loadu_si256((const __m256i *)qh);
            __m256i acc = _mm256_setzero_si256();
            for (int g = 0; g < 4; g++) {                     // grupo de 32 pesos = sub-blocos de 16 2·(4hh+g) e +1
                const __m256i lo = g < 2 ? _mm256_and_si256(g ? l1 : l0, m4)
                                         : _mm256_and_si256(_mm256_srli_epi16(g == 2 ? l0 : l1, 4), m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(hb, 2 * g), m2);
                const __m256i u = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
                const int G = 4 * hh + g, j = 2 * G;
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * G));
                const __m256i s = _mm256_set_m128i(_mm_set1_epi16(sc[j + 1]), _mm_set1_epi16(sc[j]));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, v8), _mm256_slli_epi16(s, k + 1)));
                if (k) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(refine32(k, planes, G), v8),
                                                                     _mm256_slli_epi16(s, 1)));
            }
            isum += hsum_i32(acc);
        }
        sum += y[b].d * (d * inv * (float)isum);
    }
    return sum;
}

// k constante em cada chamada: o compilador desenrola o laço dos planos.
float strata_q4kr_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    switch (k) {
    case 0: return q4kr_dot(0, x, y, n);
    case 1: return q4kr_dot(1, x, y, n);
    case 2: return q4kr_dot(2, x, y, n);
    case 3: return q4kr_dot(3, x, y, n);
    default: return q4kr_dot(k, x, y, n);
    }
}

float strata_q6kr_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    switch (k) {
    case 1: return q6kr_dot(1, x, y, n);
    case 2: return q6kr_dot(2, x, y, n);
    case 3: return q6kr_dot(3, x, y, n);
    default: return q6kr_dot(k, x, y, n);
    }
}
