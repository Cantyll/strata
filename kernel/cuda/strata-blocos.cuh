#pragma once
// STRATA: blocos montados na carga (kernel/strata_q4kr.h no repositório do STRATA), como o ggml-cuda os enxerga.
//   Q4_K_P1: o mesmo arranjo do block_q5_K; u = 2·q4 + r (5 bits)
//   Q4_K_P2: d, dmin, escalas/mínimos do Q4_K, qs[128] (4 bits baixos de u), qh[64] (2 bits altos); u = 4·q4 + r
//   Q6_K_P1: o bloco Q6_K_R1 (Q6_K com ql, qh, escalas e d, mais um plano de 1 bit por peso); w = 2q + r
//   Q6_K_P2: w = 4q + r em int8 na ordem natural dos pesos, 16 escalas int8, d
// Valor: Q4_K_P<k> = d·sc·(2u + 1 − 2^k)/2^(k+1) − dmin·mn; Q6_K_P<k> = d·sc·(2w + 1 − 2^k)/2^(k+1).

#include "common.cuh"

#include <type_traits>

struct block_strata_q4kp2 {
    half2   dm;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
    uint8_t qh[QK_K/4];
};
static_assert(sizeof(block_strata_q4kp2) == 208, "tamanho do bloco STRATA_Q4_K_P2");

struct block_strata_q6kp1 {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  sc[QK_K/16];
    half    d;
    uint8_t r[QK_K/8];
};
static_assert(sizeof(block_strata_q6kp1) == 242, "tamanho do bloco STRATA_Q6_K_P1");

struct block_strata_q6kp2 {
    int8_t w[QK_K];
    int8_t sc[QK_K/16];
    half   d;
};
static_assert(sizeof(block_strata_q6kp2) == 274, "tamanho do bloco STRATA_Q6_K_P2");
static_assert(sizeof(block_q5_K) == 176, "STRATA_Q4_K_P1 usa o arranjo do block_q5_K");

#define STRATA_PASSO(K)  (1.0f / (float) (1 << ((K) + 1)))      // 1/2^(k+1)

// Q6_K_P1: w + 64 = 2u + r (de 0 a 127; u = q + 32 é o código do Q6_K) do peso i, na ordem do q6_at de strata_q4kr.c
static __device__ __forceinline__ int strata_q6kp1_wb(const block_strata_q6kp1 * x, const int i) {
    const int h = i / 128, g = (i % 128) / 32, l = i % 32;
    const int b = x->ql[64*h + 32*(g & 1) + l];
    const int u = (g < 2 ? b & 0xF : b >> 4) | (((x->qh[32*h + l] >> (2*g)) & 3) << 4);
    return 2*u + ((x->r[i >> 3] >> (i & 7)) & 1);
}
#define STRATA_MEIO(K)   ((float) (1 - (1 << (K))))              // 1 − 2^k

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P1> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_q5_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P2> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_strata_q4kp2);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P1> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_strata_q6kp1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P2> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_strata_q6kp2);
};

// ---- Sub-células por tensor (Q4_K_D<k>, Q6_K_D<k>): mesmos blocos dos tipos P, mais uma tabela de 16 bytes do tensor
// (kernel/strata_q4kr.h). Valor: Q4_K_D<k> = d·sc·(c + δ/2^(k+4)) − dmin·mn, Q6_K_D<k> = d·sc·(q + δ/2^(k+4)), com
// δ = tab[r + 2^k·classe] (classe 0 = menor código, 2 = maior, 1 = os outros). A tabela vem do op_params do tensor de
// pesos (o carregador do llama.cpp a põe lá) e chega aos kernels por valor: no MMVQ pelo campo strata_tab dos argumentos
// de fusão; na desquantização pelo strata_cuda_tab_atual, que o ggml_cuda_mul_mat preenche antes de cada multiplicação.

#define STRATA_TAB_MARCA 0x53

// w[0..3]: a tabela de 16 bytes dos tipos D, ou a do livro (abaixo); w[4..7] só no STRATA_LIVRO1.
struct strata_tab { uint32_t w[8]; };

static constexpr __host__ __device__ bool strata_tipo_d(ggml_type t) {
    return t == GGML_TYPE_STRATA_Q4_K_D1 || t == GGML_TYPE_STRATA_Q4_K_D2 ||
           t == GGML_TYPE_STRATA_Q6_K_D1 || t == GGML_TYPE_STRATA_Q6_K_D2;
}

// Livro de códigos aninhado (STRATA_LIVRO0/1, kernel/strata_q4kr.h): bloco do IQ4_XS com a tabela de valores do tensor.
// No op_params vêm 16 (LIVRO0) ou 32 (LIVRO1) int8 e a marca no byte 32. O strata_tab que chega aos kernels já vem
// arrumado: LIVRO0, w[0..3] = T0; LIVRO1, w[0..3] = entradas pares de T1 (T1[2q]) e w[4..7] = ímpares (T1[2q + 1]),
// para o índice 2q + f sair de duas consultas de 16 pelo mesmo q e uma escolha pelo bit f.
#define STRATA_LIVRO_MARCA     0x4C
#define STRATA_LIVRO_MARCA_POS 32

static constexpr __host__ __device__ bool strata_tipo_livro(ggml_type t) {
    return t == GGML_TYPE_STRATA_LIVRO0 || t == GGML_TYPE_STRATA_LIVRO1;
}

// Tipos D com tabela "linha" (as três classes iguais: o deslocamento só depende dos bits dos níveis). Não existem no ggml
// nem em arquivo: são números livres do enum que o MMVQ usa para escolher, por tensor, um kernel sem índice de classe
// (strata_tipo_mmvq). Os blocos são os mesmos dos tipos D.
static constexpr ggml_type GGML_TYPE_STRATA_Q4_K_D1_LINHA = (ggml_type) 57;
static constexpr ggml_type GGML_TYPE_STRATA_Q4_K_D2_LINHA = (ggml_type) 58;
static constexpr ggml_type GGML_TYPE_STRATA_Q6_K_D1_LINHA = (ggml_type) 59;
static constexpr ggml_type GGML_TYPE_STRATA_Q6_K_D2_LINHA = (ggml_type) 60;
static_assert(GGML_TYPE_COUNT <= 57, "STRATA: os tipos internos LINHA colidem com tipos do ggml");

static constexpr __host__ __device__ bool strata_tipo_linha(ggml_type t) {
    return t == GGML_TYPE_STRATA_Q4_K_D1_LINHA || t == GGML_TYPE_STRATA_Q4_K_D2_LINHA ||
           t == GGML_TYPE_STRATA_Q6_K_D1_LINHA || t == GGML_TYPE_STRATA_Q6_K_D2_LINHA;
}

// tipos que levam tabela no op_params (D e livro)
static constexpr __host__ __device__ bool strata_com_tab(ggml_type t) {
    return strata_tipo_d(t) || strata_tipo_livro(t) || strata_tipo_linha(t);
}

struct block_strata_livro1 {
    block_iq4_xs b;
    uint32_t     plano[QK_K/32];   // bit j da palavra ib = bit do nível 1 do peso 32ib + j
};
static_assert(sizeof(block_iq4_xs) == 136, "STRATA_LIVRO0 usa o arranjo do block_iq4_xs");
static_assert(sizeof(block_strata_livro1) == 168, "tamanho do bloco STRATA_LIVRO1");

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_LIVRO0> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
    static constexpr int bs = sizeof(block_iq4_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_LIVRO1> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
    static constexpr int bs = sizeof(block_strata_livro1);
};

// Tabela da multiplicação em curso (lado do host) e o intervalo de memória do tensor dono dela, para a desquantização
// conferir que a tabela é a do tensor que ela está lendo.
struct strata_tab_host { strata_tab tab; const char * ini; const char * fim; };
inline thread_local strata_tab_host strata_cuda_tab_atual = {};

static strata_tab strata_tab_de(const ggml_tensor * t) {
    strata_tab tab = {};
    const int8_t * op = (const int8_t *) t->op_params;
    if (strata_tipo_livro(t->type)) {
        if (op[STRATA_LIVRO_MARCA_POS] != STRATA_LIVRO_MARCA) {
            GGML_ABORT("STRATA: o tensor %s (tipo %s) chegou sem a tabela do livro", t->name, ggml_type_name(t->type));
        }
        int8_t arr[32];
        for (int q = 0; q < 16; ++q) {
            if (t->type == GGML_TYPE_STRATA_LIVRO0) {
                arr[q] = arr[16 + q] = op[q];
            } else {
                arr[q] = op[2*q];
                arr[16 + q] = op[2*q + 1];
            }
        }
        memcpy(tab.w, arr, sizeof(arr));
        return tab;
    }
    memcpy(tab.w, op, 16);
    if ((int8_t) (tab.w[3] >> 24) != STRATA_TAB_MARCA) {
        GGML_ABORT("STRATA: o tensor %s (tipo %s) chegou sem a tabela de sub-células", t->name, ggml_type_name(t->type));
    }
    return tab;
}

static void strata_cuda_tab_definir(const ggml_tensor * src0) {
    if (!strata_com_tab(src0->type)) {
        return;
    }
    strata_cuda_tab_atual.tab = strata_tab_de(src0);
    strata_cuda_tab_atual.ini = (const char *) src0->data;
    strata_cuda_tab_atual.fim = (const char *) src0->data + ggml_nbytes(src0);
}

static strata_tab strata_cuda_tab_para(const void * vx) {
    const char * p = (const char *) vx;
    GGML_ASSERT(p >= strata_cuda_tab_atual.ini && p < strata_cuda_tab_atual.fim && "STRATA: tabela de outro tensor");
    return strata_cuda_tab_atual.tab;
}

static __device__ __forceinline__ int strata_tab_1(const strata_tab & t, const int idx) {
    return (int) (int8_t) (t.w[idx >> 2] >> (8 * (idx & 3)));
}

template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_D1> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P1> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_D2> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P2> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_D1> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P1> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_D2> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P2> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_D1_LINHA> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P1> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_D2_LINHA> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q4_K_P2> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_D1_LINHA> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P1> {};
template<>
struct ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_D2_LINHA> : ggml_cuda_type_traits<GGML_TYPE_STRATA_Q6_K_P2> {};

// 8 índices de 4 bits (q4: nibbles baixos = pesos 0..3, altos = pesos 4..7 do grupo) → os 8 bytes da tabela de 16
// guardada em w[o..o+3], como o get_int_from_table_16 do ggml (x: nibbles baixos, y: altos).
static __device__ __forceinline__ int2 strata_tab_16(const strata_tab & t, const int o, const int q4) {
    const uint32_t q = (uint32_t) q4;
    const uint32_t sel = 0x32103210 | ((q & 0x88888888) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t lo = __byte_perm(t.w[o + 0], t.w[o + 1], q >> (16*i));
        const uint32_t hi = __byte_perm(t.w[o + 2], t.w[o + 3], q >> (16*i));
        tmp[i] = __byte_perm(lo, hi, sel >> (16*i));
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// __byte_perm sem o "& 0x7777" que o compilador põe no seletor: só para seletores com o bit 3 de cada nibble apagado
// (no prmt do PTX, esse bit pede o sinal do byte em vez do byte).
static __device__ __forceinline__ uint32_t strata_prmt(const uint32_t a, const uint32_t b, const uint32_t s) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP)
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
#else
    return __byte_perm(a, b, s);
#endif
}

// 8 índices de 0 a 7 (nibbles, na ordem do strata_tab_16) → os 8 bytes da tabela de 8 em w[0..1]: uma consulta por metade.
static __device__ __forceinline__ int2 strata_tab_8(const strata_tab & t, const int q4) {
    const uint32_t q = (uint32_t) q4;
    const uint32_t a = strata_prmt(t.w[0], t.w[1], q);
    const uint32_t b = strata_prmt(t.w[0], t.w[1], q >> 16);
    return make_int2(__byte_perm(a, b, 0x6420), __byte_perm(a, b, 0x7531));
}

// MMVQ dos tipos D (strata-vecdot.cuh): no lugar de δ, o kernel usa o desvio e = δ − 8·(2r + 1 − 2^k) do filho em relação
// ao centro da sub-célula do tipo P, no índice r + 2^k·c', com c' = 0, 1 e 3 para as classes 0, 1 e 2 (o kernel tira os
// dois bits de c' de duas somas, sem comparar). Com as tabelas dos gravadores (|δ| ≤ 93) o desvio cabe em int8.
static strata_tab strata_tab_d_mmvq(const int k, const int8_t * d16, const char * nome) {
    int8_t e[16] = {};
    for (int cl = 0; cl < 3; ++cl) {
        for (int r = 0; r < (1 << k); ++r) {
            const int v = d16[r + (cl << k)] - 8*(2*r + 1 - (1 << k));
            if (v < -128 || v > 127) {
                GGML_ABORT("STRATA: a tabela do tensor %s sai do alcance do kernel (desvio %d)", nome, v);
            }
            e[r + ((cl == 2 ? 3 : cl) << k)] = (int8_t) v;
        }
    }
    strata_tab tab = {};
    memcpy(tab.w, e, sizeof(e));
    return tab;
}

static int strata_d_k(const ggml_type t) {
    return t == GGML_TYPE_STRATA_Q4_K_D1 || t == GGML_TYPE_STRATA_Q6_K_D1 ? 1 : 2;
}

// a tabela δ do tensor D tem as três classes iguais?
static bool strata_tab_e_linha(const ggml_tensor * t) {
    if (!strata_tipo_d(t->type)) {
        return false;
    }
    const strata_tab tab = strata_tab_de(t);
    const int8_t * d = (const int8_t *) tab.w;
    const int k = strata_d_k(t->type);
    for (int r = 0; r < (1 << k); ++r) {
        if (d[r] != d[r + (1 << k)] || d[r] != d[r + (2 << k)]) {
            return false;
        }
    }
    return true;
}

// O tipo que o MMVQ instancia para o tensor (e o gate, na fusão): o D com tabela linha vai para o kernel LINHA, se os dois
// forem linha; senão o próprio tipo.
static ggml_type strata_tipo_mmvq(const ggml_tensor * t, const ggml_tensor * gate) {
    if (!strata_tab_e_linha(t) || (gate && !strata_tab_e_linha(gate))) {
        return t->type;
    }
    switch (t->type) {
        case GGML_TYPE_STRATA_Q4_K_D1: return GGML_TYPE_STRATA_Q4_K_D1_LINHA;
        case GGML_TYPE_STRATA_Q4_K_D2: return GGML_TYPE_STRATA_Q4_K_D2_LINHA;
        case GGML_TYPE_STRATA_Q6_K_D1: return GGML_TYPE_STRATA_Q6_K_D1_LINHA;
        default:                       return GGML_TYPE_STRATA_Q6_K_D2_LINHA;
    }
}

// A tabela que o MMVQ recebe para o tipo tipo_mmvq: a do livro como está; a dos tipos D em desvios (strata_tab_d_mmvq);
// a dos tipos LINHA com os desvios só da classe do meio: k = 1, w[0] = e(0) e w[1] = e(1) − e(0) (inteiros, o desvio é
// afim em r); k = 2, os 4 bytes e(r) em w[0].
static strata_tab strata_tab_mmvq(const ggml_tensor * t, const ggml_type tipo_mmvq) {
    const strata_tab tab = strata_tab_de(t);
    if (!strata_tipo_d(t->type)) {
        return tab;
    }
    const int k = strata_d_k(t->type);
    const strata_tab e = strata_tab_d_mmvq(k, (const int8_t *) tab.w, t->name);
    if (!strata_tipo_linha(tipo_mmvq)) {
        return e;
    }
    const int8_t * e8 = (const int8_t *) e.w + (1 << k);   // classe do meio
    strata_tab l = {};
    if (k == 1) {
        l.w[0] = (uint32_t) (int32_t) e8[0];
        l.w[1] = (uint32_t) (int32_t) (e8[1] - e8[0]);
    } else {
        memcpy(l.w, e8, 4);
    }
    return l;
}

// 4 bits (os 4 baixos de b) → 0xFF nos bytes cujo bit está ligado
static __device__ __forceinline__ uint32_t strata_mascara_4(const uint32_t b) {
    return (((b & 0xF) * 0x00204081u) & 0x01010101u) * 0xFFu;
}
