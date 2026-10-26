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

// Os tipos R, P e D fazem mais conta por byte que os k-quants e, com os pesos vindo da RAM, o prefetch do hardware não
// acompanha (medido: geração do nível 1 D 2,3× mais lenta que o tipo P com os pesos fora do cache, 1,8× dentro).
// Pedir o bloco STRATA_PF à frente resolve.
#ifndef STRATA_PF
#define STRATA_PF 8
#endif
#define STRATA_PREFETCH(x, bb) do { for (int l_ = 0; l_ < (bb); l_ += 64) _mm_prefetch((const char *)(x) + STRATA_PF*(bb) + l_, _MM_HINT_T0); } while (0)

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

// Soma dos 8 floats de v.
static inline float hsum_f32(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_movehdup_ps(h)));
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
        STRATA_PREFETCH(x, bb);
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
        STRATA_PREFETCH(x, bb);
        // escalas em int16, a mesma metade nas duas lanes de 128 bits; Σ sc_j·bsum_j (cabe em 32 bits) vezes off
        const __m128i s8 = _mm_loadu_si128((const __m128i *)sc);
        const __m256i sw = _mm256_cvtepi8_epi16(s8);
        const int64_t isum0 = (int64_t)off * hsum_i32(_mm256_madd_epi16(sw, _mm256_loadu_si256((const __m256i *)y[b].bsums)));
        const __m256i sA = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sw));
        const __m256i sB = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(sw, 1));
        int64_t isum = isum0;
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
                // lane baixa: escala j; lane alta: j + 1 (j = 2G; as 8 primeiras vêm de sA, as 8 últimas de sB)
                const int m = j % 8;
                const __m256i s = _mm256_shuffle_epi8(G < 4 ? sA : sB,
                    _mm256_setr_epi8(2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                                     2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                                     2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3,
                                     2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3));
                if (k == 1) {                                 // 2u + r cabe em 7 bits: um maddubs só
                    const __m256i u7 = _mm256_add_epi8(_mm256_add_epi8(u, u), refine32(1, planes, G));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u7, v8), _mm256_slli_epi16(s, 1)));
                    continue;
                }
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

// ---- Blocos montados na carga (Q4_K_P<k>, Q6_K_P<k>) ----
// O índice u (ou w) de cada peso já está inteiro no bloco: um maddubs por 32 pesos, como no Q5_K/Q6_K do ggml.
// Σ t·q8 = 2·Σ u·q8 + (1 − 2^k)·Σ q8, e o termo das bsums sai uma vez por bloco.
// Diferente dos tipos R, aqui a soma inteira de cada bloco vai para um acumulador float de 8 posições (como no ggml),
// então o float final pode diferir da referência escalar nos últimos bits (muda a ordem e o arredondamento das somas
// em float; erro relativo da ordem de 1e-7). Ganho medido: ~7% no nível 1.

// Escalas/mínimos do Q4_K desembrulhados (como no ggml) e Σ sc·bsum, Σ mn·bsum.
static inline __attribute__((always_inline)) __m256i q4k_scales(const uint8_t *scales, const strata_block_q8_K *y,
                                                                int32_t *sum_c, int32_t *sum_m) {
    uint32_t ut[4];
    memcpy(ut, scales, 12);
    ut[3] = ((ut[2] >> 4) & 0x0f0f0f0f) | (((ut[1] >> 6) & 0x03030303) << 4);
    const uint32_t ua = ut[1] & 0x3f3f3f3f;
    ut[1] = (ut[2] & 0x0f0f0f0f) | (((ut[0] >> 6) & 0x03030303) << 4);
    ut[2] = ua;
    ut[0] &= 0x3f3f3f3f;
    const __m128i sm = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)ut));
    const __m128i mm = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(ut + 2)));
    const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128((const __m128i *)y->bsums), _mm_loadu_si128((const __m128i *)(y->bsums + 8)));
    __m128i h2 = _mm_hadd_epi32(_mm_madd_epi16(sm, bs), _mm_madd_epi16(mm, bs));
    h2 = _mm_hadd_epi32(h2, h2);
    *sum_c = _mm_cvtsi128_si32(h2);
    *sum_m = _mm_extract_epi32(h2, 1);
    return _mm256_broadcastsi128_si256(sm);
}

static inline __attribute__((always_inline)) float q4kp_dot(const int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q4kp_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    const __m256i m4 = _mm256_set1_epi8(0xF), mh = _mm256_set1_epi8(k == 1 ? 1 : 3);
    float sum = 0;
    __m256 accf = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        const uint8_t *qs = x + (k == 1 ? 48 : 16), *qh = x + (k == 1 ? 16 : 144);
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, bb);
        int32_t sum_c, sum_m;
        const __m256i sc256 = q4k_scales(x + 4, &y[b], &sum_c, &sum_m);
        __m256i hb = _mm256_loadu_si256((const __m256i *)qh);
        __m256i acc = _mm256_setzero_si256();
        for (int h = 0; h < 4; h++) {
            if (k == 2 && h == 2) hb = _mm256_loadu_si256((const __m256i *)(qh + 32));
            const __m256i q4b = _mm256_loadu_si256((const __m256i *)(qs + 32 * h));
            for (int s = 0; s < 2; s++) {
                const int j = 2 * h + s, sh = k == 1 ? j : 2 * (j % 4);
                const __m256i lo = s ? _mm256_and_si256(_mm256_srli_epi16(q4b, 4), m4) : _mm256_and_si256(q4b, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(hb, sh), mh);
                const __m256i u = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
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

// Q6_K_P2: w com sinal, então |w|·sinal(q8, w) no maddubs (como o Q8_0 do ggml); |−128| = 128 sem sinal, ainda cabe.
static inline __attribute__((always_inline)) float q6kp_dot(const int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 1));
    float sum = 0;
    __m256 accf = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(k)) {
        const float d = fp16_to_fp32((uint16_t)(x[272] | (x[273] << 8)));
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, strata_q6kp_block_bytes(k));
        const __m256i sw = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 256)));
        int64_t isum = (int64_t)(1 - (1 << k)) * hsum_i32(_mm256_madd_epi16(sw, _mm256_loadu_si256((const __m256i *)y[b].bsums)));
        const __m256i sA = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sw));
        const __m256i sB = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(sw, 1));
        __m256i acc = _mm256_setzero_si256();                 // cada posição: |soma| < 8·2·(2·128·127)·128 < 2^27
        for (int hh = 0; hh < 2; hh++) {
            for (int g = 0; g < 4; g++) {
                const int G = 4 * hh + g, m = (2 * G) % 8;
                const __m256i w = _mm256_loadu_si256((const __m256i *)(x + 32 * G));
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * G));
                const __m256i s = _mm256_shuffle_epi8(hh ? sB : sA,
                    _mm256_setr_epi8(2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                                     2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                                     2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3,
                                     2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3));
                const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(v8, w));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, s));
            }
        }
        const float dd = y[b].d * d * inv;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(2 * dd), _mm256_cvtepi32_ps(acc), accf);
        sum += dd * (float)isum;
    }
    return sum + hsum_f32(accf);
}

// u (de 0 a 63, o código do Q6_K mais 32) dos 32 pesos do grupo g (0 a 3) de uma metade de 128 pesos, com l0 e l1 os
// 64 bytes de ql e hb os 32 de qh da metade, na ordem de q6_at (strata_q4kr.c).
static inline __attribute__((always_inline)) __m256i q6k_u32(__m256i l0, __m256i l1, __m256i hb, int g) {
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3);
    const __m256i lo = g < 2 ? _mm256_and_si256(g ? l1 : l0, m4) : _mm256_and_si256(_mm256_srli_epi16(g == 2 ? l0 : l1, 4), m4);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(hb, 2 * g), m2);
    return _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
}

// Escalas j = 2G e 2G + 1 do Q6_K (int16 em sA para G < 4, sB para G ≥ 4), uma em cada lane de 128 bits.
static inline __attribute__((always_inline)) __m256i q6k_s2(__m256i sA, __m256i sB, int G) {
    const int m = (2 * G) % 8;
    return _mm256_shuffle_epi8(G >= 4 ? sB : sA,
        _mm256_setr_epi8(2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                         2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                         2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3,
                         2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3));
}

// Q6_K_P1 (= Q6_K_R1): w + 64 = 2u + r vai de 0 a 127, sem sinal, num maddubs só; Σ t·q8 = 2·Σ (w + 64)·q8 − 129·Σ q8.
static float q6kp1_dot(const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    float sum = 0;
    __m256 accf = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(1)) {
        const float d = fp16_to_fp32((uint16_t)(x[208] | (x[209] << 8)));
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, strata_q6kp_block_bytes(1));
        const uint8_t *planes = x + STRATA_Q6K_BYTES;
        const __m256i sw = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 192)));
        const int64_t isum = -129 * (int64_t)hsum_i32(_mm256_madd_epi16(sw, _mm256_loadu_si256((const __m256i *)y[b].bsums)));
        const __m256i sA = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sw));
        const __m256i sB = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(sw, 1));
        __m256i acc = _mm256_setzero_si256();
        for (int hh = 0; hh < 2; hh++) {
            const __m256i l0 = _mm256_loadu_si256((const __m256i *)(x + 64 * hh));
            const __m256i l1 = _mm256_loadu_si256((const __m256i *)(x + 64 * hh + 32));
            const __m256i hb = _mm256_loadu_si256((const __m256i *)(x + 128 + 32 * hh));
            for (int g = 0; g < 4; g++) {
                const int G = 4 * hh + g;
                const __m256i u = q6k_u32(l0, l1, hb, g);
                const __m256i w7 = _mm256_add_epi8(_mm256_add_epi8(u, u), refine32(1, planes, G));
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * G));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(w7, v8), q6k_s2(sA, sB, G)));
            }
        }
        const float dd = y[b].d * d * 0.25f;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(2 * dd), _mm256_cvtepi32_ps(acc), accf);
        sum += dd * (float)isum;
    }
    return sum + hsum_f32(accf);
}

float strata_q4kp_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    return k == 1 ? q4kp_dot(1, x, y, n) : q4kp_dot(2, x, y, n);
}

float strata_q6kp_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    return k == 1 ? q6kp1_dot(x, y, n) : q6kp_dot(2, x, y, n);
}

// ---- Sub-células por tensor (Q4_K_D<k> e Q6_K_D<k>) ----
// Σ v·q8 = d·Σ sc·(Σ c·q8 + Σ δ·q8 / 2^(k+4)) − dmin·Σ mn·Σ q8: dois maddubs por grupo de 32 pesos, um com o código c
// (como no Q4_K) e um com δ, que sai de um pshufb na tabela (16 bytes) com índice r + 2^k·classe; no Q4_K a classe sai
// de outro pshufb (indexado por c), no Q6_K de dois cmpeq (c = menor código, c = maior). δ tem sinal, então |δ|·sinal(q8, δ) no maddubs (|δ| ≤ 93, sem saturar).
// O resultado em float pode diferir da referência nos últimos bits, como nos tipos P.

static inline __attribute__((always_inline)) float q4kd_dot(const int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q4kp_block_bytes(k);
    const __m256i m4 = _mm256_set1_epi8(0xF), mh = _mm256_set1_epi8(k == 1 ? 1 : 3), mr = _mm256_set1_epi8((1 << k) - 1);
    const __m256i mc = _mm256_set1_epi8(k == 1 ? 7 : 3);
    const __m256i T = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)tab));
    // 2^k·classe de cada código c (0 a 15): 0 no menor, 2^(k+1) no maior, 2^k nos outros
    const char K1 = (char)(1 << k), K2 = (char)(2 << k);
    const __m256i CL = _mm256_setr_epi8(0, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K2,
                                        0, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K1, K2);
    const float inv = 1.0f / (float)(1 << (k + 4));
    float sum = 0;
    __m256 accc = _mm256_setzero_ps(), accd = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        const uint8_t *qs = x + (k == 1 ? 48 : 16), *qh = x + (k == 1 ? 16 : 144);
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, bb);
        int32_t sum_c, sum_m;
        const __m256i sc256 = q4k_scales(x + 4, &y[b], &sum_c, &sum_m);
        __m256i hb = _mm256_loadu_si256((const __m256i *)qh);
        __m256i ac = _mm256_setzero_si256(), ad = _mm256_setzero_si256();
        for (int h = 0; h < 4; h++) {
            if (k == 2 && h == 2) hb = _mm256_loadu_si256((const __m256i *)(qh + 32));
            const __m256i q4b = _mm256_loadu_si256((const __m256i *)(qs + 32 * h));
            for (int s = 0; s < 2; s++) {
                const int j = 2 * h + s, sh = k == 1 ? j : 2 * (j % 4);
                const __m256i lo = s ? _mm256_and_si256(_mm256_srli_epi16(q4b, 4), m4) : _mm256_and_si256(q4b, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(hb, sh), mh);
                const __m256i c = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, k), mc), _mm256_slli_epi16(hi, 4 - k));
                const __m256i dl = _mm256_shuffle_epi8(T, _mm256_add_epi8(_mm256_and_si256(lo, mr), _mm256_shuffle_epi8(CL, c)));
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * j));
                const __m256i s16 = scale16(sc256, j);
                ac = _mm256_add_epi32(ac, _mm256_madd_epi16(_mm256_maddubs_epi16(c, v8), s16));
                ad = _mm256_add_epi32(ad, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(dl), _mm256_sign_epi8(v8, dl)), s16));
            }
        }
        const float dd = y[b].d * d;
        accc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(ac), accc);
        accd = _mm256_fmadd_ps(_mm256_set1_ps(dd * inv), _mm256_cvtepi32_ps(ad), accd);
        sum -= y[b].d * dmin * (float)sum_m;
    }
    return sum + hsum_f32(accc) + hsum_f32(accd);
}

// Q6_K_D: c = q = (w + 2^(k+5)) >> k − 32 (deslocamento lógico de 16 bits com máscara: não há srai de 8 bits) e r = w & (2^k − 1).
static inline __attribute__((always_inline)) float q6kd_dot(const int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {   // k = 2
    const __m256i T = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)tab));
    const __m256i mr = _mm256_set1_epi8((1 << k) - 1), K2 = _mm256_set1_epi8(1 << k), off = _mm256_set1_epi8((char)(1 << (k + 5)));
    const __m256i cm = _mm256_set1_epi8((char)(0xFF >> k)), c32 = _mm256_set1_epi8(32), cmn = _mm256_set1_epi8(-32), cmx = _mm256_set1_epi8(31);
    const float inv = 1.0f / (float)(1 << (k + 4));
    __m256 accc = _mm256_setzero_ps(), accd = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(k)) {
        const float d = fp16_to_fp32((uint16_t)(x[272] | (x[273] << 8)));
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, strata_q6kp_block_bytes(k));
        const __m256i sw = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 256)));
        const __m256i sA = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sw));
        const __m256i sB = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(sw, 1));
        __m256i ac = _mm256_setzero_si256(), ad = _mm256_setzero_si256();
        for (int G = 0; G < 8; G++) {
            const int m = (2 * G) % 8;
            const __m256i w = _mm256_loadu_si256((const __m256i *)(x + 32 * G));
            const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * G));
            const __m256i s = _mm256_shuffle_epi8(G >= 4 ? sB : sA,
                _mm256_setr_epi8(2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                                 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1, 2 * m, 2 * m + 1,
                                 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3,
                                 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3, 2 * m + 2, 2 * m + 3));
            const __m256i c = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(_mm256_add_epi8(w, off), k), cm), c32);
            const __m256i idx = _mm256_add_epi8(_mm256_add_epi8(_mm256_and_si256(w, mr), K2),
                _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpeq_epi8(c, cmx), K2), _mm256_and_si256(_mm256_cmpeq_epi8(c, cmn), K2)));
            const __m256i dl = _mm256_shuffle_epi8(T, idx);
            ac = _mm256_add_epi32(ac, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(v8, c)), s));
            ad = _mm256_add_epi32(ad, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(dl), _mm256_sign_epi8(v8, dl)), s));
        }
        const float dd = y[b].d * d;
        accc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(ac), accc);
        accd = _mm256_fmadd_ps(_mm256_set1_ps(dd * inv), _mm256_cvtepi32_ps(ad), accd);
    }
    return hsum_f32(accc) + hsum_f32(accd);
}

float strata_q4kd_vec_dot_q8_K_avx2(int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    return k == 1 ? q4kd_dot(1, tab, x, y, n) : q4kd_dot(2, tab, x, y, n);
}

// Q6_K_D1 sobre o bloco Q6_K_R1: c = u − 32 com u sem sinal (o −32 vai pelas bsums), r do plano.
static float q6kd1_dot(const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const __m256i T = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)tab));
    const __m256i K2 = _mm256_set1_epi8(2), z = _mm256_setzero_si256(), c63 = _mm256_set1_epi8(63);
    float sum = 0;
    __m256 accc = _mm256_setzero_ps(), accd = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(1)) {
        const float d = fp16_to_fp32((uint16_t)(x[208] | (x[209] << 8)));
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, strata_q6kp_block_bytes(1));
        const uint8_t *planes = x + STRATA_Q6K_BYTES;
        const __m256i sw = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(x + 192)));
        const int32_t sbs = hsum_i32(_mm256_madd_epi16(sw, _mm256_loadu_si256((const __m256i *)y[b].bsums)));
        const __m256i sA = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sw));
        const __m256i sB = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(sw, 1));
        __m256i ac = _mm256_setzero_si256(), ad = _mm256_setzero_si256();
        for (int hh = 0; hh < 2; hh++) {
            const __m256i l0 = _mm256_loadu_si256((const __m256i *)(x + 64 * hh));
            const __m256i l1 = _mm256_loadu_si256((const __m256i *)(x + 64 * hh + 32));
            const __m256i hb = _mm256_loadu_si256((const __m256i *)(x + 128 + 32 * hh));
            for (int g = 0; g < 4; g++) {
                const int G = 4 * hh + g;
                const __m256i u = q6k_u32(l0, l1, hb, g);
                const __m256i idx = _mm256_add_epi8(_mm256_add_epi8(refine32(1, planes, G), K2),
                    _mm256_sub_epi8(_mm256_and_si256(_mm256_cmpeq_epi8(u, c63), K2), _mm256_and_si256(_mm256_cmpeq_epi8(u, z), K2)));
                const __m256i dl = _mm256_shuffle_epi8(T, idx);
                const __m256i v8 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * G));
                const __m256i s = q6k_s2(sA, sB, G);
                ac = _mm256_add_epi32(ac, _mm256_madd_epi16(_mm256_maddubs_epi16(u, v8), s));
                ad = _mm256_add_epi32(ad, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(dl), _mm256_sign_epi8(v8, dl)), s));
            }
        }
        const float dd = y[b].d * d;
        accc = _mm256_fmadd_ps(_mm256_set1_ps(dd), _mm256_cvtepi32_ps(ac), accc);
        accd = _mm256_fmadd_ps(_mm256_set1_ps(dd * (1.0f / 32)), _mm256_cvtepi32_ps(ad), accd);
        sum -= 32 * dd * (float)sbs;
    }
    return sum + hsum_f32(accc) + hsum_f32(accd);
}

float strata_q6kd_vec_dot_q8_K_avx2(int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    return k == 1 ? q6kd1_dot(tab, x, y, n) : q6kd_dot(2, tab, x, y, n);
}

// ---- Livro de códigos aninhado (STRATA_LIVRO0 e STRATA_LIVRO1) ----
// O produto do IQ4_XS do ggml com a tabela do tensor no lugar de kvalues_iq4nl. No LIVRO1 o índice é 2q + f: as entradas
// pares e ímpares de T1 viram duas tabelas de 16 (TP[q] = T1[2q], TI[q] = T1[2q + 1]), dois pshufb pelo mesmo q e um
// blendv pela máscara do plano (bit j da palavra do sub-bloco → byte j em 0xFF).
static inline __attribute__((always_inline)) __m256i livro_mascara(uint32_t p) {
    const __m256i sel = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int)p),
        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    return _mm256_cmpeq_epi8(_mm256_and_si256(b, sel), sel);
}

static inline __attribute__((always_inline)) float livro_dot(const int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_livro_block_bytes(k);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256i TP, TI;
    if (k == 0) {
        TP = TI = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)tab));
    } else {
        const __m128i pi = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        const __m128i a = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)tab), pi);
        const __m128i c = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(tab + 16)), pi);
        TP = _mm256_broadcastsi128_si256(_mm_unpacklo_epi64(a, c));
        TI = _mm256_broadcastsi128_si256(_mm_unpackhi_epi64(a, c));
    }
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const int8_t *q8 = y[b].qs;
        STRATA_PREFETCH(x, bb);
        uint32_t sh = (uint32_t)(x[2] | (x[3] << 8));
        uint32_t pl[8];
        if (k == 1) memcpy(pl, x + STRATA_LIVRO0_BYTES, 32);
        __m256i s1 = _mm256_setzero_si256(), s2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib += 2) {
            const __m128i b1 = _mm_loadu_si128((const __m128i *)(x + 8 + 16 * ib));
            const __m128i b2 = _mm_loadu_si128((const __m128i *)(x + 24 + 16 * ib));
            const __m256i q1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(b1, 4), m4), _mm_and_si128(b1, m4));
            const __m256i q2 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(b2, 4), m4), _mm_and_si128(b2, m4));
            __m256i v1 = _mm256_shuffle_epi8(TP, q1), v2 = _mm256_shuffle_epi8(TP, q2);
            if (k == 1) {
                v1 = _mm256_blendv_epi8(v1, _mm256_shuffle_epi8(TI, q1), livro_mascara(pl[ib]));
                v2 = _mm256_blendv_epi8(v2, _mm256_shuffle_epi8(TI, q2), livro_mascara(pl[ib + 1]));
            }
            const __m256i y1 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * ib));
            const __m256i y2 = _mm256_loadu_si256((const __m256i *)(q8 + 32 * ib + 32));
            const __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(v1), _mm256_sign_epi8(y1, v1));
            const __m256i p2 = _mm256_maddubs_epi16(_mm256_abs_epi8(v2), _mm256_sign_epi8(y2, v2));
            const int16_t ls1 = (int16_t)(((x[4 + ib / 2] & 0xF) | ((sh << 4) & 0x30)) - 32);
            const int16_t ls2 = (int16_t)(((x[4 + ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32);
            sh >>= 4;
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(p1, _mm256_set1_epi16(ls1)));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(p2, _mm256_set1_epi16(ls2)));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * y[b].d), _mm256_cvtepi32_ps(_mm256_add_epi32(s1, s2)), acc);
    }
    return hsum_f32(acc);
}

float strata_livro_vec_dot_q8_K_avx2(int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    return k == 0 ? livro_dot(0, tab, x, y, n) : livro_dot(1, tab, x, y, n);
}
