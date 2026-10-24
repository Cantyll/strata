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

// ---- Blocos montados na carga (Q4_K_P<k> e Q6_K_P<k>; layout em strata_q4kr.h) ----

// u (de 0 a 2^(k+4) − 1) do peso i de um bloco Q4_K_P<k>.
static inline int q4kp_u(int k, const uint8_t *x, int i) {
    const int j = i / 32, l = i % 32;
    if (k == 1) return q4_at(x + 48, i) | (((x[16 + l] >> j) & 1) << 4);
    return q4_at(x + 16, i) | (((x[144 + 32 * (j / 4) + l] >> (2 * (j % 4))) & 3) << 4);
}

void strata_q4kr_to_q4kp(int k, const uint8_t *src, uint8_t *dst, int64_t n) {
    const int sb = strata_q4kr_block_bytes(k), db = strata_q4kp_block_bytes(k);
    for (int64_t b = 0; b < n / STRATA_QK; b++, src += sb, dst += db) {
        memset(dst, 0, db);
        memcpy(dst, src, 16);                                  // d, dmin, escalas e mínimos
        uint8_t *qs = dst + (k == 1 ? 48 : 16), *qh = dst + (k == 1 ? 16 : 144);
        for (int i = 0; i < STRATA_QK; i++) {
            const int u = (q4_at(src + 16, i) << k) | refine_at(k, src + STRATA_Q4K_BYTES, i);
            const int j = i / 32, l = i % 32;
            qs[(i / 64) * 32 + l] |= (uint8_t)((u & 0xF) << ((i % 64) < 32 ? 0 : 4));
            if (k == 1) qh[l] |= (uint8_t)((u >> 4) << j);
            else qh[32 * (j / 4) + l] |= (uint8_t)((u >> 4) << (2 * (j % 4)));
        }
    }
}

void strata_q4kp_dequantize_row(int k, const uint8_t *x, float *y, int64_t n) {
    const int bb = strata_q4kp_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min(j, x + 4, &sc, &mn);
            const float a = d * sc, m = dmin * mn;
            for (int l = 0; l < 32; l++) {
                const int i = 32 * j + l, t = 2 * q4kp_u(k, x, i) + 1 - (1 << k);
                y[b * STRATA_QK + i] = a * ((float)t * inv) - m;
            }
        }
    }
}

float strata_q4kp_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q4kp_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 1));
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        int32_t sum_t = 0, sum_m = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min(j, x + 4, &sc, &mn);
            int32_t s = 0;
            for (int l = 0; l < 32; l++) {
                const int i = 32 * j + l;
                s += (2 * q4kp_u(k, x, i) + 1 - (1 << k)) * y[b].qs[i];
            }
            sum_t += sc * s;
            sum_m += mn * (y[b].bsums[2 * j] + y[b].bsums[2 * j + 1]);
        }
        sum += y[b].d * (d * inv * (float)sum_t - dmin * (float)sum_m);
    }
    return sum;
}

void strata_q6kr_to_q6kp(int k, const uint8_t *src, uint8_t *dst, int64_t n) {
    const int sb = strata_q6kr_block_bytes(k);
    if (k == 1) {                                              // Q6_K_P1 = Q6_K_R1
        memcpy(dst, src, (size_t)(n / STRATA_QK) * sb);
        return;
    }
    for (int64_t b = 0; b < n / STRATA_QK; b++, src += sb, dst += strata_q6kp_block_bytes(k)) {
        for (int i = 0; i < STRATA_QK; i++)
            dst[i] = (uint8_t)(int8_t)(q6_at(src, i) * (1 << k) + refine_at(k, src + STRATA_Q6K_BYTES, i));
        memcpy(dst + 256, src + 192, 18);                      // escalas e d
    }
}

// w = 2^k·q + r do peso i, d e as escalas de um bloco Q6_K_P<k>.
static inline int q6kp_w(int k, const uint8_t *x, int i) {
    return k == 1 ? 2 * q6_at(x, i) + refine_at(1, x + STRATA_Q6K_BYTES, i) : (int8_t)x[i];
}
static inline float q6kp_d(int k, const uint8_t *x) {
    const int o = k == 1 ? 208 : 272;
    return fp16_to_fp32((uint16_t)(x[o] | (x[o + 1] << 8)));
}
static inline const int8_t *q6kp_sc(int k, const uint8_t *x) { return (const int8_t *)(x + (k == 1 ? 192 : 256)); }

void strata_q6kp_dequantize_row(int k, const uint8_t *x, float *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 1));
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(k)) {
        const float d = q6kp_d(k, x);
        const int8_t *sc = q6kp_sc(k, x);
        for (int i = 0; i < STRATA_QK; i++) {
            const float a = d * sc[i / 16];
            const int t = 2 * q6kp_w(k, x, i) + 1 - (1 << k);
            y[b * STRATA_QK + i] = a * ((float)t * inv);
        }
    }
}

float strata_q6kp_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 1));
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(k)) {
        const float d = q6kp_d(k, x);
        const int8_t *sc = q6kp_sc(k, x);
        int64_t isum = 0;
        for (int j = 0; j < 16; j++) {
            int32_t s = 0;
            for (int l = 0; l < 16; l++) s += (2 * q6kp_w(k, x, 16 * j + l) + 1 - (1 << k)) * y[b].qs[16 * j + l];
            isum += (int64_t)sc[j] * s;
        }
        sum += y[b].d * (d * inv * (float)isum);
    }
    return sum;
}

// ---- Sub-células por tensor (Q4_K_D<k> e Q6_K_D<k>; tabela em strata_q4kr.h) ----

void strata_tab_padrao(int k, int8_t *tab) {
    memset(tab, 0, STRATA_TAB_BYTES);
    for (int cl = 0; cl < 3; cl++)
        for (int r = 0; r < (1 << k); r++) tab[r + (cl << k)] = (int8_t)(8 * (2 * r + 1 - (1 << k)));
    tab[STRATA_TAB_BYTES - 1] = STRATA_TAB_MARCA;
}

static inline int tab_idx(int k, int c, int cmin, int cmax, int r) {
    return r + ((c == cmin ? 0 : c == cmax ? 2 : 1) << k);
}

void strata_q4kd_dequantize_row(int k, const int8_t *tab, const uint8_t *x, float *y, int64_t n) {
    const int bb = strata_q4kp_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 4));
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min(j, x + 4, &sc, &mn);
            const float a = d * sc, m = dmin * mn;
            for (int l = 0; l < 32; l++) {
                const int i = 32 * j + l, u = q4kp_u(k, x, i), c = u >> k;
                const int t = (c << (k + 4)) + tab[tab_idx(k, c, 0, 15, u & ((1 << k) - 1))];
                y[b * STRATA_QK + i] = a * ((float)t * inv) - m;
            }
        }
    }
}

// Σ sc·Σ c·q8 e Σ sc·Σ δ·q8 em inteiros, juntados em float: 2^(k+4)·Σ c·q8 passaria de 32 bits com k = 2.
float strata_q4kd_vec_dot_q8_K(int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_q4kp_block_bytes(k);
    const float inv = 1.0f / (float)(1 << (k + 4));
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        const float dmin = fp16_to_fp32((uint16_t)(x[2] | (x[3] << 8)));
        int32_t sum_c = 0, sum_d = 0, sum_m = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t sc, mn;
            get_scale_min(j, x + 4, &sc, &mn);
            int32_t sc_ = 0, sd = 0;
            for (int l = 0; l < 32; l++) {
                const int i = 32 * j + l, u = q4kp_u(k, x, i), c = u >> k;
                sc_ += c * y[b].qs[i];
                sd += tab[tab_idx(k, c, 0, 15, u & ((1 << k) - 1))] * y[b].qs[i];
            }
            sum_c += sc * sc_;
            sum_d += sc * sd;
            sum_m += mn * (y[b].bsums[2 * j] + y[b].bsums[2 * j + 1]);
        }
        sum += y[b].d * (d * ((float)sum_c + (float)sum_d * inv) - dmin * (float)sum_m);
    }
    return sum;
}

void strata_q6kd_dequantize_row(int k, const int8_t *tab, const uint8_t *x, float *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 4));
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(k)) {
        const float d = q6kp_d(k, x);
        const int8_t *sc = q6kp_sc(k, x);
        for (int i = 0; i < STRATA_QK; i++) {
            const int w = q6kp_w(k, x, i), q = w >> k, t = q * (1 << (k + 4)) + tab[tab_idx(k, q, -32, 31, w & ((1 << k) - 1))];
            y[b * STRATA_QK + i] = d * sc[i / 16] * ((float)t * inv);
        }
    }
}

float strata_q6kd_vec_dot_q8_K(int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const float inv = 1.0f / (float)(1 << (k + 4));
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += strata_q6kp_block_bytes(k)) {
        const float d = q6kp_d(k, x);
        const int8_t *sc = q6kp_sc(k, x);
        int64_t ic = 0, id = 0;
        for (int j = 0; j < 16; j++) {
            int32_t a = 0, e = 0;
            for (int l = 0; l < 16; l++) {
                const int i = 16 * j + l, w = q6kp_w(k, x, i), q = w >> k;
                a += q * y[b].qs[i];
                e += tab[tab_idx(k, q, -32, 31, w & ((1 << k) - 1))] * y[b].qs[i];
            }
            ic += (int64_t)sc[j] * a;
            id += (int64_t)sc[j] * e;
        }
        sum += y[b].d * (d * ((float)ic + (float)id * inv));
    }
    return sum;
}

// ---- Livro de códigos aninhado (STRATA_LIVRO0 e STRATA_LIVRO1; layout em strata_q4kr.h) ----

static inline int livro_ls(const uint8_t *x, int ib) {
    const int sh = x[2] | (x[3] << 8);
    return (((x[4 + ib / 2] >> 4 * (ib % 2)) & 0xF) | (((sh >> 2 * ib) & 3) << 4)) - 32;
}

// índice na tabela do peso j (0 a 31) do sub-bloco ib
static inline int livro_idx(int k, const uint8_t *x, int ib, int j) {
    const uint8_t b = x[8 + 16 * ib + (j % 16)];
    const int q = j < 16 ? (b & 0xF) : (b >> 4);
    if (k == 0) return q;
    const uint8_t *p = x + STRATA_LIVRO0_BYTES + 4 * ib;
    return 2 * q + ((p[j / 8] >> (j % 8)) & 1);
}

void strata_livro_dequantize_row(int k, const int8_t *tab, const uint8_t *x, float *y, int64_t n) {
    const int bb = strata_livro_block_bytes(k);
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        for (int ib = 0; ib < 8; ib++) {
            const float dl = d * (float)livro_ls(x, ib);
            for (int j = 0; j < 32; j++) y[b * STRATA_QK + 32 * ib + j] = dl * (float)tab[livro_idx(k, x, ib, j)];
        }
    }
}

// Σ ls·Σ T·q8 em inteiros por bloco (|Σ| < 2^31: 256·127·127·32), como o IQ4_XS.
float strata_livro_vec_dot_q8_K(int k, const int8_t *tab, const uint8_t *x, const strata_block_q8_K *y, int64_t n) {
    const int bb = strata_livro_block_bytes(k);
    float sum = 0;
    for (int64_t b = 0; b < n / STRATA_QK; b++, x += bb) {
        const float d = fp16_to_fp32((uint16_t)(x[0] | (x[1] << 8)));
        int32_t s = 0;
        for (int ib = 0; ib < 8; ib++) {
            int32_t a = 0;
            for (int j = 0; j < 32; j++) a += tab[livro_idx(k, x, ib, j)] * y[b].qs[32 * ib + j];
            s += livro_ls(x, ib) * a;
        }
        sum += d * y[b].d * (float)s;
    }
    return sum;
}
