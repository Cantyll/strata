// Referência escalar dos tipos Q4_K_R<k> e Q6_K_R<k>. Ver strata_q4kr.h para o layout e a fórmula.
// Compilar sem contração de FMA (-ffp-contract=off) para o resultado ser o mesmo em qualquer máquina.
#include "strata_q4kr.h"
#include <math.h>
#include <string.h>

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {                                   // subnormal: normaliza
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | ((exp + 112) << 23) | (man << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
    else bits = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Igual a get_scale_min_k4 do ggml: escala e mínimo de 6 bits do sub-bloco j.
static void get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// q4 do peso i (ordem do Q4_K): grupos de 64 pesos em 32 bytes, nibble baixo = 32 primeiros, alto = 32 seguintes.
static inline int q4_at(const uint8_t *qs, int i) {
    const uint8_t b = qs[(i / 64) * 32 + (i % 32)];
    return (i % 64) < 32 ? (b & 0xF) : (b >> 4);
}

static inline int refine_at(int k, const uint8_t *planes, int i) {
    int r = 0;
    for (int p = 0; p < k; p++) r = (r << 1) | ((planes[32 * p + i / 8] >> (i % 8)) & 1);
    return r;
}

void strata_q4kr_dequantize_row(int k, const uint8_t *x, float *y, int64_t n) {
    const int bb = strata_q4kr_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));            // potência de 2: exata
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        const uint8_t *scales = x + 4, *qs = x + 16, *planes = x + STRATA_Q4K_BYTES;
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min(j, scales, &sc, &mn);
            const float a = d * sc, m = dmin * mn;
            for (int l = 0; l < 32; l++) {
                const int i = 32 * j + l;
                const int t = (q4_at(qs, i) << (k + 1)) + 2 * refine_at(k, planes, i) + 1 - (1 << k);
                y[b * STRATA_QK + i] = a * ((float)t * inv) - m;
            }
        }
    }
}

void strata_q8_K_quantize_row(const float *x, strata_block_q8_K *y, int64_t n) {
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += STRATA_QK) {
        float amax = 0;
        for (int i = 0; i < STRATA_QK; i++) amax = fmaxf(amax, fabsf(x[i]));
        strata_block_q8_K *o = &y[b];
        if (amax == 0) { memset(o, 0, sizeof(*o)); continue; }
        const float iscale = 127.0f / amax;
        for (int i = 0; i < STRATA_QK; i++) {
            int v = (int)lrintf(iscale * x[i]);
            o->qs[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
        }
        for (int g = 0; g < STRATA_QK / 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++) s += o->qs[16 * g + i];
            o->bsums[g] = (int16_t)s;
        }
        o->d = 1.0f / iscale;
    }
}

float strata_q4kr_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q4kr_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        const uint8_t *scales = x + 4, *qs = x + 16, *planes = x + STRATA_Q4K_BYTES;
        int32_t sum_t = 0, sum_m = 0;                          // Σ sc_j·Σ t·q8   e   Σ mn_j·Σ q8
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min(j, scales, &sc, &mn);
            int32_t s = 0;
            for (int l = 0; l < 32; l++) {
                const int i = 32 * j + l;
                const int t = (q4_at(qs, i) << (k + 1)) + 2 * refine_at(k, planes, i) + 1 - (1 << k);
                s += t * y[b].qs[i];
            }
            sum_t += sc * s;
            sum_m += mn * (y[b].bsums[2 * j] + y[b].bsums[2 * j + 1]);
        }
        sum += y[b].d * (d * inv * (float)sum_t - dmin * (float)sum_m);
    }
    return sum;
}

// q do peso i na ordem do Q6_K (de −32 a 31): metades de 128 pesos; em cada uma, 4 grupos de 32 que tiram o nibble
// de ql[l] ou ql[l+32] (baixo nos grupos 0 e 1, alto nos 2 e 3) e 2 bits de qh[l].
static inline int q6_at(const uint8_t *x, int i) {
    const int h = i / 128, g = (i % 128) / 32, l = i % 32;
    const uint8_t *ql = x + 64 * h, *qh = x + 128 + 32 * h;
    const uint8_t b = ql[l + 32 * (g & 1)];
    const int lo = g < 2 ? (b & 0xF) : (b >> 4);
    return (lo | (((qh[l] >> (2 * g)) & 3) << 4)) - 32;
}

void strata_q6kr_dequantize_row(int k, const uint8_t *x, float *y, int64_t n) {
    const int bb = strata_q6kr_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[208] | (x[209] << 8)));
        const int8_t *sc = (const int8_t *)(x + 192);
        const uint8_t *planes = x + STRATA_Q6K_BYTES;
        for (int i = 0; i < STRATA_QK; i++) {
            const float a = d * sc[i / 16];
            const int t = q6_at(x, i) * (1 << (k + 1)) + 2 * refine_at(k, planes, i) + 1 - (1 << k);
            y[b * STRATA_QK + i] = a * ((float)t * inv);
        }
    }
}

float strata_q6kr_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q6kr_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[208] | (x[209] << 8)));
        const int8_t *sc = (const int8_t *)(x + 192);
        const uint8_t *planes = x + STRATA_Q6K_BYTES;
        int64_t isum = 0;                                      // com k = 3 o pior caso passa de 2^31
        for (int j = 0; j < 16; j++) {
            int32_t s = 0;
            for (int l = 0; l < 16; l++) {
                const int i = 16 * j + l;
                const int t = q6_at(x, i) * (1 << (k + 1)) + 2 * refine_at(k, planes, i) + 1 - (1 << k);
                s += t * y[b].qs[i];
            }
            isum += (int64_t)sc[j] * s;
        }
        sum += y[b].d * (d * inv * (float)isum);
    }
    return sum;
}
