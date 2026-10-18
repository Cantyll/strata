// Tipo Q4_K_R<k> do STRATA: o bloco Q4_K do estrato 0 seguido de k planos de 1 bit por peso (níveis 1..k do
// modo residual). Versão de referência, escalar, em C99: define o resultado que as versões SIMD e CUDA
// precisam reproduzir. Especificação: docs/02-especificacao-formato.md, B.8.
//
// Bloco de 256 pesos, 144 + 32k bytes (k = 1: 5,5 bits/peso; k = 2: 6,5; k = 3: 7,5):
//   bytes   0..143          o bloco Q4_K, byte a byte (d, dmin, 12 bytes de escalas/mínimos, 128 bytes de índices)
//   bytes 144 + 32(p-1)..   plano do nível p (p = 1..k): bit i (byte i/8, bit i%8) = bit do nível p do peso i
// O peso i é o i-ésimo na ordem do Q4_K: sub-bloco j = i/32, posição l = i%32.
// Subir do nível k para k+1 é acrescentar um plano de 32 bytes no fim do bloco; nada antes dele muda.
//
// Valor do peso (centro da sub-célula), com a = d·sc_j e m = dmin·mn_j como no Q4_K:
//   r = bits dos níveis 1..k (nível 1 é o mais significativo),  t = 2^(k+1)·q4 + 2r + 1 − 2^k
//   v = a · t / 2^(k+1) − m
// O mínimo é o do Q4_K, sem requantizar; o deslocamento da sub-célula está no t.
//
// Tipo Q6_K_R<k>: o mesmo para as matrizes que o Q4_K_M guarda em Q6_K. Bloco de 256 pesos, 210 + 32k bytes:
//   bytes   0..209          o bloco Q6_K, byte a byte (ql[128], qh[64], 16 escalas int8, d)
//   bytes 210 + 32(p-1)..   plano do nível p, como no Q4_K_R; peso i na ordem do Q6_K (sub-bloco de 16 j = i/16)
// Com a = d·sc_j (sc com sinal) e q de −32 a 31:  t = 2^(k+1)·q + 2r + 1 − 2^k,  v = a · t / 2^(k+1).
// A sub-célula r conta no sentido de q crescente; com sc < 0 isso é o sentido de valor decrescente.
#ifndef STRATA_Q4KR_H
#define STRATA_Q4KR_H
#include <stdint.h>

#define STRATA_QK 256
#define STRATA_Q4K_BYTES 144

#define STRATA_Q6K_BYTES 210

static inline int strata_q4kr_block_bytes(int k) { return STRATA_Q4K_BYTES + 32 * k; }
static inline int strata_q6kr_block_bytes(int k) { return STRATA_Q6K_BYTES + 32 * k; }

// Ativações quantizadas como o Q8_K do llama.cpp: 256 valores int8 com uma escala float e somas de 16 em 16.
typedef struct {
    float d;
    int8_t qs[STRATA_QK];
    int16_t bsums[STRATA_QK / 16];
} strata_block_q8_K;

// n = número de pesos (múltiplo de 256); k = 1, 2 ou 3.
void strata_q4kr_dequantize_row(int k, const uint8_t *x, float *y, int64_t n);
void strata_q8_K_quantize_row(const float *x, strata_block_q8_K *y, int64_t n);
// Produto escalar de uma linha Q4_K_R<k> com uma linha Q8_K, em inteiros por sub-bloco como os k-quants.
float strata_q4kr_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);

void strata_q6kr_dequantize_row(int k, const uint8_t *x, float *y, int64_t n);
float strata_q6kr_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);

// Versões AVX2 (strata_q4kr_avx2.c): o mesmo resultado, bit a bit.
float strata_q4kr_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);
float strata_q6kr_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);

// Blocos montados na carga (tipos Q4_K_P<k> e Q6_K_P<k>, "prontos"): o arquivo continua em camadas (base + planos),
// mas ao carregar o modelo cada bloco R é convertido uma vez num bloco em que o índice inteiro de cada peso já está
// junto, como nos k-quants. Os valores são os mesmos do tipo R, bit a bit; só muda o arranjo dos bits na RAM.
// Implementação: strata_q4kr.c (escalar) e strata_q4kr_avx2.c.
//
// Q4_K_P<k>, k = 1 ou 2, com u = 2^k·q4 + r (de 0 a 2^(k+4) − 1) e t = 2u + 1 − 2^k (o mesmo t do Q4_K_R<k>):
//   k = 1, 176 bytes (5,5 bits/peso), arranjo idêntico ao block_q5_K do ggml:
//     d, dmin, 12 bytes de escalas/mínimos (os do Q4_K), qh[32], qs[128]
//     qs como no Q4_K, com os 4 bits baixos de u; qh[l] bit j = bit 4 de u do peso 32j + l
//   k = 2, 208 bytes (6,5 bits/peso):
//     d, dmin, 12 bytes de escalas/mínimos, qs[128] (4 bits baixos de u, como no Q4_K), qh[64]
//     qh[32·(j/4) + l] bits 2·(j%4) e 2·(j%4)+1 = bits 4 e 5 de u do peso 32j + l
// Q6_K_P<k>, k = 1 ou 2, 274 bytes, com w = 2^k·q + r em int8 (de −64 a 63, ou de −128 a 127) e t = 2w + 1 − 2^k:
//   w[256] (ordem do Q6_K), 16 escalas int8, d.  Com k = 2 o tamanho é o mesmo do Q6_K_R2 (8,56 bits/peso);
//   com k = 1 a RAM gasta 1 bit/peso a mais que o arquivo nessas matrizes.
#define STRATA_Q6KP_BYTES 274
static inline int strata_q4kp_block_bytes(int k) { return k == 1 ? 176 : 208; }

// Montagem: n pesos de blocos R<k> em src para blocos P<k> em dst.
void strata_q4kr_to_q4kp(int k, const uint8_t *src, uint8_t *dst, int64_t n);
void strata_q6kr_to_q6kp(int k, const uint8_t *src, uint8_t *dst, int64_t n);
void strata_q4kp_dequantize_row(int k, const uint8_t *x, float *y, int64_t n);
float strata_q4kp_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);
void strata_q6kp_dequantize_row(int k, const uint8_t *x, float *y, int64_t n);
float strata_q6kp_vec_dot_q8_K(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);
float strata_q4kp_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);
float strata_q6kp_vec_dot_q8_K_avx2(int k, const uint8_t *x, const strata_block_q8_K *y, int64_t n);

#endif
