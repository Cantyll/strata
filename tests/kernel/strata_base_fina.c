/* Base "fina" do modo residual: em vez de quantizar o Q4_K e depois subdividir cada célula em 2^T, ajusta as escalas e os
 * mínimos de cada sub-bloco direto na grade de 4 + T bits (com a imatrix, se houver) e tira o Q4_K por corte: q4 = u >> T,
 * planos = os T bits baixos de u. Os níveis 0 … T continuam aninhados e o bloco Q4_K resultante é um Q4_K comum.
 *
 * Valor no nível T (o mesmo de strata_q4kr.h): d·sc·(2u + 1 − 2^T)/2^(T+1) − dmin·mn, u em 0 … 2^(4+T) − 1.
 * Com s = d·sc/2^T (passo fino) e M = dmin·mn + s·(2^T − 1)/2, fica s·u − M: a forma que make_qkx3_quants ajusta.
 *
 * make_qkx3_quants e make_qp_quants são cópias do ggml-quants.c do llama.cpp (MIT), commit 2a66703, para que a busca
 * seja a mesma do quantize_row_q4_K_impl; o resto segue a mesma função, trocando 15 por 2^(4+T) − 1.
 *
 * Modo conjunto (strata_base_fina_conjunta(1), a "escala conjunta" de docs/07): além da busca na grade fina, faz a busca
 * de sempre na grade de 4 bits (a do Q4_K_M) e testa 11 pontos na reta entre as duas soluções (escala e mínimo), ficando
 * com o de menor erro somado dos níveis com os pesos dados. Com o peso só no nível 0, a escolha é a do Q4_K_M; com o peso
 * nos níveis de cima, anda em direção à base fina.
 *
 *   cc -O2 -shared -fPIC tests/kernel/strata_base_fina.c -lm -o libstrata_base_fina.so
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/* fp16 <-> fp32, só o necessário (arredondamento ao par mais próximo, como o GGML_FP32_TO_FP16 em C puro) */
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000;
        int sh = 14 - e;
        uint32_t h = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return sign | h;
    }
    uint32_t h = (e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return sign | h;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
    if (e == 0) {
        if (!m) x = sign;
        else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; x = sign | (e << 23) | (m << 13); }
    } else if (e == 31) x = sign | 0x7f800000 | (m << 13);
    else x = sign | ((e + 127 - 15) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

/* Erro de um candidato (escala, mínimo, índices finos) somado nos níveis 0 … T, cada um com o seu peso: o nível j usa o
 * centro do grupo de 2^(T−j) índices finos que contém L[i]. Sem pesos por nível (pn == NULL), só o nível T. */
static _Thread_local int g_T = 0;            /* por thread: o modo conjunto troca g_pn no meio da busca */
static _Thread_local const float * g_pn = NULL;
static int g_conj = 0;

void strata_base_fina_conjunta(int on) { g_conj = on; }

static float erro_niveis(int n, const float * x, const float * weights, const uint8_t * L, float scale, float min) {
    float tot = 0;
    for (int j = 0; j <= g_T; ++j) {
        const float pj = g_pn ? g_pn[j] : (j == g_T ? 1.f : 0.f);
        if (pj == 0) continue;
        const int sh = g_T - j;
        const float meio = 0.5f * (float)((1 << sh) - 1);
        float e = 0;
        for (int i = 0; i < n; ++i) {
            float diff = scale * ((float)((L[i] >> sh) << sh) + meio) + min - x[i];
            float w = weights ? weights[i] : x[i]*x[i];
            e += w * diff*diff;
        }
        tot += pj * e;
    }
    return tot;
}

static float make_qkx3_quants(int n, int nmax, const float * x, const float * weights, uint8_t * L, float * the_min,
                              uint8_t * Laux, float rmin, float rdelta, int nstep, int use_mad) {
    float min = x[0], max = x[0];
    float sum_w = weights ? weights[0] : x[0]*x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max <= min) { memset(L, 0, n); *the_min = -min; return 0.f; }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff*diff;
        float w = weights ? weights[i] : x[i]*x[i];
        best_mad += w * diff;
    }
    if (g_pn) best_mad = erro_niveis(n, x, weights, L, scale, min);
    if (nstep < 1) { *the_min = -min; return scale; }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            const float v = iscale*(x[i] - min);
            const int l = v > 0 ? nearest_int(MIN(v, nmax)) : 0;
            Laux[i] = l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) { this_min = 0; this_scale = sum_xl / sum_l2; }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff*diff;
                float w = weights ? weights[i] : x[i]*x[i];
                mad += w * diff;
            }
            if (g_pn) mad = erro_niveis(n, x, weights, Laux, this_scale, this_min);
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_mad = mad; scale = this_scale; min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float make_qp_quants(int n, int nmax, const float * x, uint8_t * L, const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) max = MAX(max, x[i]);
    if (max < GROUP_MAX_EPS) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) L[i] = nearest_int(iscale * x[i]);
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) { float diff = x[i] - scale*L[i]; best_mse += quant_weights[i]*diff*diff; }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max, scale_is = 1/iscale_is, mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = MIN(nmax, nearest_int(iscale_is*x[i]));
            float diff = x[i] - scale_is*l;
            mse += quant_weights[i]*diff*diff;
        }
        if (mse < best_mse) { best_mse = mse; iscale = iscale_is; }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = MIN(nmax, nearest_int(iscale * x[i]));
        L[i] = l;
        sumlx += quant_weights[i]*x[i]*l;
        suml2 += quant_weights[i]*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = MIN(nmax, nearest_int(x[i] * sl2 / slx));
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) { L[i] = new_l; sumlx = slx; suml2 = sl2; ++n_changed; }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

/* Uma linha de n_per_row pesos. out: blocos Q4_K (144 B); u: o índice fino de cada peso (0 … 2^(4+T) − 1), de onde saem
 * q4 = u >> T e os planos. quant_weights: a imatrix da linha (uma por coluna), ou NULL. */
void strata_base_fina_q4k(const float * x, uint8_t * out, uint8_t * u, int64_t n_per_row, const float * quant_weights, int T) {
    const int nmax = (1 << (4 + T)) - 1;
    const float meio = 0.5f * (float)((1 << T) - 1);
    uint8_t Laux[32], Ls[QK_K/32], Lm[QK_K/32];
    float weights[32], sw[QK_K/32], mins[QK_K/32], scales[QK_K/32];
    for (int64_t i = 0; i < n_per_row / QK_K; i++, x += QK_K, out += 144, u += QK_K) {
        float sum_x2 = 0;
        for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
        float sigma2 = 2*sum_x2/QK_K, av_x = sqrtf(sigma2);
        for (int j = 0; j < QK_K/32; ++j) {
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*i + 32*j;
                for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32*j + l]*x[32*j + l]);
            } else {
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            }
            float sumw = 0;
            for (int l = 0; l < 32; ++l) sumw += weights[l];
            sw[j] = sumw;
            float M, s = make_qkx3_quants(32, nmax, x + 32*j, weights, u + 32*j, &M, Laux, -0.9f, 0.05f, 36, 0);
            if (g_conj && T > 0) {
                /* candidato do Q4: busca de 4 bits, convertida para a grade fina (mesmos centros de grupo) */
                const float * pn = g_pn;
                uint8_t Lc[32];
                g_pn = NULL;
                float Mc, sc = make_qkx3_quants(32, 15, x + 32*j, weights, Lc, &Mc, Laux, -0.9f, 0.05f, 36, 0);
                g_pn = pn;
                const float sa = sc / (float)(1 << T), Ma = Mc + sa * meio;
                /* o mínimo do Q4_K é M − s·meio e não pode ser negativo: cada candidato é avaliado já com ele preso em 0,
                   do jeito que vai ser gravado (a busca fina sozinha não sabe disso) */
                float best = INFINITY, bs = s, bM = MAX(M, s * meio);
                for (int it = 0; it <= 10 && (sa > 0 || s > 0); ++it) {
                    const float t = 0.1f * it, st = (1 - t) * sa + t * s;
                    if (st <= 0) continue;
                    const float Mt = MAX((1 - t) * Ma + t * M, st * meio);
                    for (int l = 0; l < 32; ++l) Laux[l] = MAX(0, MIN(nmax, nearest_int((x[32*j + l] + Mt) / st)));
                    float e = erro_niveis(32, x + 32*j, weights, Laux, st, -Mt);
                    if (e < best) { best = e; bs = st; bM = Mt; }
                }
                s = bs; M = bM;
            }
            scales[j] = s * (float)(1 << T);                  /* passo da célula do Q4 */
            mins[j] = MAX(0.f, M - s * meio);                 /* mínimo do Q4_K (sem sinal) */
        }
        float d_block = make_qp_quants(QK_K/32, 63, scales, Ls, sw);
        float m_block = make_qp_quants(QK_K/32, 63, mins, Lm, sw);
        uint8_t * sc12 = out + 4;
        memset(sc12, 0, 12);
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = Ls[j], lm = Lm[j];
            if (j < 4) { sc12[j] = ls; sc12[j+4] = lm; }
            else { sc12[j+4] = (ls & 0xF) | ((lm & 0xF) << 4); sc12[j-4] |= ((ls >> 4) << 6); sc12[j-0] |= ((lm >> 4) << 6); }
        }
        uint16_t hd = f32_to_f16(d_block), hm = f32_to_f16(m_block);
        memcpy(out, &hd, 2); memcpy(out + 2, &hm, 2);
        const float d = f16_to_f32(hd), dmin = f16_to_f32(hm);
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t sc, m;
            get_scale_min_k4(j, sc12, &sc, &m);
            const float dc = d * sc, dm = dmin * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = dc ? nearest_int((x[32*j + ii] + dm) / dc * (float)(1 << T) + meio) : 0;
                u[32*j + ii] = MAX(0, MIN(nmax, l));
            }
        }
        uint8_t * q = out + 16;
        for (int j = 0; j < QK_K; j += 64, q += 32)
            for (int l = 0; l < 32; ++l) q[l] = (u[j + l] >> T) | ((u[j + l + 32] >> T) << 4);
    }
}

/* pesos_nivel: T + 1 pesos do erro de cada nível na busca da escala (NULL = só o nível T). */
void strata_base_fina_q4k_rows(const float * x, uint8_t * out, uint8_t * u, int64_t nrows, int64_t n_per_row,
                               const float * quant_weights, int T, const float * pesos_nivel) {
    g_T = T;
    g_pn = pesos_nivel;
    for (int64_t r = 0; r < nrows; ++r)
        strata_base_fina_q4k(x + r*n_per_row, out + r*(n_per_row/QK_K)*144, u + r*n_per_row, n_per_row, quant_weights, T);
}
