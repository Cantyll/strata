// Layout separado (tipos S) da precisão sob demanda (docs/08-precisao-sob-demanda.md, seção 3 e experimento E2).
//
// A matriz no nível k fica em k + 1 buffers: a base (os blocos Q4_K ou Q6_K do GGUF, byte a byte, que podem ficar
// mapeados direto do arquivo publicado) e um buffer por nível p = 1..k com 32 bytes por bloco de 256 pesos (o plano do
// nível p, bit i = byte i/8, bit i%8, na ordem do bloco da base). São os mesmos bytes do tipo R (B.8), só que o plano de
// cada nível mora num buffer próprio: subir um nível é ler mais um buffer, descer é soltar o último, e o nível 0 é a
// base sozinha com o kernel do k-quant.
//
// Na RAM, cada plano fica na "ordem qh" (byte l, bit j = peso 32j + l do bloco, a ordem do qh do Q5_K), que o kernel lê
// com um deslocamento e um and; strata_plano_para_qh converte do arranjo do arquivo ao subir de nível.
// O valor de cada peso é o do tipo R<k>; o produto escalar soma como os tipos P (acumulador float de 8 posições), então
// o float final pode diferir da referência escalar do tipo R nos últimos bits (erro relativo da ordem de 1e-7).
#ifndef STRATA_SEPARADO_H
#define STRATA_SEPARADO_H
#include "strata_q4kr.h"

#define STRATA_PLANO_BYTES 32

// Converte nblocos planos de 32 bytes do arranjo do arquivo (bit i = byte i/8, bit i%8) para a ordem qh.
void strata_plano_para_qh(const uint8_t *src, uint8_t *dst, int64_t nblocos);
// planos[p - 1] = buffer do nível p, na ordem qh; k = 0, 1 ou 2. n = número de pesos (múltiplo de 256).
float strata_q4ks_vec_dot_q8_K_avx2(int k, const uint8_t *base, const uint8_t *const *planos, const strata_block_q8_K *y, int64_t n);
float strata_q6ks_vec_dot_q8_K_avx2(int k, const uint8_t *base, const uint8_t *const *planos, const strata_block_q8_K *y, int64_t n);

#endif
