// Imitação mínima do common.cuh do ggml-cuda para compilar os kernels do STRATA (kernel/cuda/*.cuh) como C++ comum,
// na CPU, e conferir a conta sem GPU. Só o que os cabeçalhos do STRATA usam.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define GGML_CUDA_RESTRICT

typedef int cudaStream_t;
struct dim_t { int x = 0, y = 0, z = 0; };
static dim_t blockIdx, threadIdx;

static inline float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0)       v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else              v = std::ldexp((float) (m | 0x400), (int) e - 25);
    return s ? -v : v;
}
struct half  { uint16_t b; operator float() const { return h2f(b); } };
struct half2 { half x, y; };
struct float2 { float x, y; };
struct int2 { int x, y; };
static inline int2 make_int2(int x, int y) { return { x, y }; }
static inline float  __half2float(half h)     { return h2f(h.b); }
static inline half   __low2half(half2 h)      { return h.x; }
static inline half   __high2half(half2 h)     { return h.y; }
static inline float  __low2float(half2 h)     { return h2f(h.x.b); }
static inline float2 __half22float2(half2 h)  { return { h2f(h.x.b), h2f(h.y.b) }; }

static inline int ggml_cuda_dp4a(const int a, const int b, int c) {
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}

template <typename T> static inline T ggml_cuda_cast(float x) { return (T) x; }

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
#define QR4_XS 2
#define QI4_XS (QK_K / (4*QR4_XS))

struct block_q5_K { half2 dm; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; };
struct block_q8_1 { half2 ds; int8_t qs[QK8_1]; };
struct block_iq4_xs { half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; };

enum ggml_type { GGML_TYPE_STRATA_Q4_K_P1 = 43, GGML_TYPE_STRATA_Q4_K_P2 = 44, GGML_TYPE_STRATA_Q6_K_P1 = 45, GGML_TYPE_STRATA_Q6_K_P2 = 46,
                 GGML_TYPE_STRATA_Q4_K_D1 = 47, GGML_TYPE_STRATA_Q4_K_D2 = 48, GGML_TYPE_STRATA_Q6_K_D1 = 49, GGML_TYPE_STRATA_Q6_K_D2 = 50,
                 GGML_TYPE_STRATA_LIVRO0 = 51, GGML_TYPE_STRATA_LIVRO1 = 52, GGML_TYPE_COUNT = 57 };

// o mínimo do ggml_tensor e das funções do ggml que o lado host dos cabeçalhos usa
struct ggml_tensor { ggml_type type; void * data; size_t nbytes; char name[64]; int32_t op_params[16]; };
static inline size_t ggml_nbytes(const ggml_tensor * t) { return t->nbytes; }
static inline const char * ggml_type_name(ggml_type) { return "strata"; }
#define GGML_ABORT(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); abort(); } while (0)
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "GGML_ASSERT(%s)\n", #x); abort(); } } while (0)

// intrínsecos de bytes do CUDA
static inline uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    const uint64_t v = ((uint64_t) y << 32) | x;
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) r |= (uint32_t) ((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
    return r;
}
static inline uint32_t __vcmpeq4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) if (((a >> (8 * i)) & 0xFF) == ((b >> (8 * i)) & 0xFF)) r |= 0xFFu << (8 * i);
    return r;
}
template <ggml_type T> struct ggml_cuda_type_traits;

// de vecdotq.cuh e dequantize.cuh
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;
    return x32;
}
static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
