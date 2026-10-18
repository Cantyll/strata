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
#ifndef STRATA_Q4KR_H
#define STRATA_Q4KR_H
#include <stdint.h>

#define STRATA_QK 256
#define STRATA_Q4K_BYTES 144

static inline int strata_q4kr_block_bytes(int k) { return STRATA_Q4K_BYTES + 32 * k; }

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

#endif
