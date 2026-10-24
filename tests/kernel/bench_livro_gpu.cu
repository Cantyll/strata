// Micro-benchmark na GPU (docs/09, opção A): quanto custa consultar a tabela do livro de códigos com 16, 32 e 64 valores
// num produto matriz-vetor (a geração, um token por vez), contra uma decodificação uniforme que lê exatamente os mesmos
// bytes. Só velocidade; a qualidade do livro foi medida em F16.
//
// Formato (por bloco de 32 pesos, em arrays separados para a leitura coalescer): 16 bytes de códigos de 4 bits (nibble j
// da palavra w = peso 8w + j), k planos de 32 bits (bit i = peso i), uma escala int8 por bloco e um half por 8 blocos.
// O mesmo do IQ4_XS no nível 0 (4,31 bits/peso aqui) e +1 bit por nível.
//   livro:    valor = tabela[código | plano1 << 4 | plano2 << 5], tabela int8 de 16·2^k entradas em registradores,
//             consultada com __byte_perm (como o IQ4_NL do llama.cpp): 1, 2 ou 4 consultas de 16 + seleção por byte.
//   uniforme: valor = código − 8·2^k (como Q4_0/Q5_0/Q6 sem mínimo), bits de nível encaixados nos bytes.
// Os dois fazem 8 __dp4a por bloco com a ativação em int8 e somam d·escala·Σ·escala_da_ativação.
//
//   nvcc -O3 -arch=native bench_livro_gpu.cu -o bench && ./bench
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <random>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));             \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

constexpr int LINHAS = 14336, COLS = 4096, NSB = COLS / 32, NMAT = 8;   // ffn do Llama-3.1-8B; 8 matrizes > cache L2

__constant__ uint32_t TAB[16];   // até 64 entradas int8

__device__ __forceinline__ uint32_t espalha4(uint32_t b) {      // 4 bits → bit 0 de cada nibble
    return (b & 1) | ((b & 2) << 3) | ((b & 4) << 6) | ((b & 8) << 9);
}
__device__ __forceinline__ uint32_t espalha_par(uint32_t x) {   // bits 0, 2, 4, 6 → bit 4 de cada byte
    return ((x & 1) << 4) | ((x & 4) << 10) | ((x & 16) << 16) | ((x & 64) << 22);
}

template <int K, bool LIVRO>
__global__ void mv(const uint4 *__restrict__ q, const uint32_t *__restrict__ p1, const uint32_t *__restrict__ p2,
                   const int8_t *__restrict__ sc, const half *__restrict__ d, const int *__restrict__ ax,
                   const int *__restrict__ ax_par, const float *__restrict__ ad, const int *__restrict__ as,
                   float *__restrict__ y) {
    const int linha = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (linha >= LINHAS) return;
    uint32_t t[LIVRO ? (4 << K) : 1];
    if (LIVRO)
#pragma unroll
        for (int i = 0; i < (4 << K); i++) t[i] = TAB[i];
    float acc = 0.f;
    for (int b = lane; b < NSB; b += 32) {
        const size_t i = (size_t)linha * NSB + b;
        const uint4 qq = q[i];
        const uint32_t qw[4] = {qq.x, qq.y, qq.z, qq.w};
        const uint32_t pa = K >= 1 ? p1[i] : 0, pb = K >= 2 ? p2[i] : 0;
        const int *a = (LIVRO ? ax : ax_par) + b * 8;
        int soma = 0;
#pragma unroll
        for (int w = 0; w < 4; w++) {
            if (LIVRO) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t n16 = (qw[w] >> (16 * h)) & 0xFFFF;
                    const uint32_t sel = n16 & 0x7777, hsel = 0x3210 | ((n16 & 0x8888) >> 1);
                    auto l16 = [&](int o) {
                        return __byte_perm(__byte_perm(t[o], t[o + 1], sel), __byte_perm(t[o + 2], t[o + 3], sel), hsel);
                    };
                    uint32_t r = l16(0);
                    if (K >= 1) {
                        const uint32_t s1 = 0x3210 | (espalha4((pa >> (8 * w + 4 * h)) & 0xF) << 2);
                        r = __byte_perm(r, l16(4), s1);
                        if (K >= 2) {
                            const uint32_t r2 = __byte_perm(l16(8), l16(12), s1);
                            r = __byte_perm(r, r2, 0x3210 | (espalha4((pb >> (8 * w + 4 * h)) & 0xF) << 2));
                        }
                    }
                    soma = __dp4a((int)r, a[2 * w + h], soma);
                }
            } else {
                uint32_t e = qw[w] & 0x0F0F0F0F, o = (qw[w] >> 4) & 0x0F0F0F0F;   // pesos pares e ímpares da palavra
                if (K >= 1) {
                    const uint32_t x = (pa >> (8 * w)) & 0xFF;
                    e |= espalha_par(x), o |= espalha_par(x >> 1);
                }
                if (K >= 2) {
                    const uint32_t x = (pb >> (8 * w)) & 0xFF;
                    e |= espalha_par(x) << 1, o |= espalha_par(x >> 1) << 1;
                }
                soma = __dp4a((int)e, a[2 * w], soma);
                soma = __dp4a((int)o, a[2 * w + 1], soma);
            }
        }
        if (!LIVRO) soma -= (8 << K) * as[b];
        acc += __half2float(d[i / 8]) * sc[i] * ad[b] * soma;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) y[linha] = acc;
}

struct Mat {
    uint4 *q;
    uint32_t *p1, *p2;
    int8_t *sc;
    half *d;
};

int main() {
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("GPU: %s, %d SMs, L2 %d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    std::mt19937 rng(1);
    const size_t nb = (size_t)LINHAS * NSB;
    std::vector<uint32_t> hq(nb * 4), hp1(nb), hp2(nb);
    std::vector<int8_t> hsc(nb);
    std::vector<half> hd(nb / 8);
    std::vector<Mat> M(NMAT);
    for (int m = 0; m < NMAT; m++) {
        for (auto &v : hq) v = rng();
        for (auto &v : hp1) v = rng();
        for (auto &v : hp2) v = rng();
        for (auto &v : hsc) { int s = (int)(rng() % 63) - 32; v = (int8_t)(s == 0 ? 1 : s); }
        for (auto &v : hd) v = __float2half(1e-3f * (1 + rng() % 7));
        CK(cudaMalloc(&M[m].q, nb * 16));
        CK(cudaMalloc(&M[m].p1, nb * 4));
        CK(cudaMalloc(&M[m].p2, nb * 4));
        CK(cudaMalloc(&M[m].sc, nb));
        CK(cudaMalloc(&M[m].d, nb / 8 * 2));
        CK(cudaMemcpy(M[m].q, hq.data(), nb * 16, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(M[m].p1, hp1.data(), nb * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(M[m].p2, hp2.data(), nb * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(M[m].sc, hsc.data(), nb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(M[m].d, hd.data(), nb / 8 * 2, cudaMemcpyHostToDevice));
    }
    // ativação int8 por bloco de 32 (ordem natural e pares/ímpares por palavra), escala e soma por bloco
    std::vector<int8_t> a8(COLS), a8p(COLS);
    std::vector<float> had(NSB);
    std::vector<int> has(NSB, 0);
    for (int k = 0; k < COLS; k++) a8[k] = (int8_t)((int)(rng() % 255) - 127);
    for (int b = 0; b < NSB; b++) {
        had[b] = 1e-2f * (1 + rng() % 5);
        for (int k = 0; k < 32; k++) has[b] += a8[b * 32 + k];
        for (int w = 0; w < 4; w++)
            for (int i = 0; i < 4; i++) {
                a8p[b * 32 + w * 8 + i] = a8[b * 32 + w * 8 + 2 * i];
                a8p[b * 32 + w * 8 + 4 + i] = a8[b * 32 + w * 8 + 2 * i + 1];
            }
    }
    int *ax, *axp, *as;
    float *ad, *y;
    CK(cudaMalloc(&ax, COLS));
    CK(cudaMalloc(&axp, COLS));
    CK(cudaMalloc(&ad, NSB * 4));
    CK(cudaMalloc(&as, NSB * 4));
    CK(cudaMalloc(&y, LINHAS * 4));
    CK(cudaMemcpy(ax, a8.data(), COLS, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(axp, a8p.data(), COLS, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ad, had.data(), NSB * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(as, has.data(), NSB * 4, cudaMemcpyHostToDevice));
    std::vector<int8_t> tab(64);
    for (int i = 0; i < 64; i++) tab[i] = (int8_t)std::lround(-127 + 254.0 * std::pow((i + 0.5) / 64, 1.3));
    std::sort(tab.begin(), tab.end());
    CK(cudaMemcpyToSymbol(TAB, tab.data(), 64));

    // referência na CPU: 64 linhas da última matriz carregada (hq, hp1, hp2, hsc, hd)
    auto ref = [&](int K, bool livro, int r) {
        double acc = 0;
        for (int b = 0; b < NSB; b++) {
            const size_t i = (size_t)r * NSB + b;
            long soma = 0;
            for (int k = 0; k < 32; k++) {
                const uint32_t c4 = (hq[i * 4 + k / 8] >> (4 * (k % 8))) & 0xF;
                const uint32_t c = c4 | (K >= 1 ? ((hp1[i] >> k) & 1) << 4 : 0) | (K >= 2 ? ((hp2[i] >> k) & 1) << 5 : 0);
                soma += (long)(livro ? tab[c] : (int)c - (8 << K)) * a8[b * 32 + k];
            }
            acc += (double)__half2float(hd[i / 8]) * hsc[i] * had[b] * soma;
        }
        return acc;
    };
    // banda de cópia medida, como referência do pico
    {
        void *s, *t;
        const size_t n = 512u << 20;
        CK(cudaMalloc(&s, n));
        CK(cudaMalloc(&t, n));
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0), cudaEventCreate(&e1);
        CK(cudaMemcpy(t, s, n, cudaMemcpyDeviceToDevice));
        cudaEventRecord(e0);
        for (int r = 0; r < 10; r++) cudaMemcpy(t, s, n, cudaMemcpyDeviceToDevice);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        printf("cópia na GPU: %.0f GB/s (leitura + escrita)\n\n", 2.0 * n * 10 / ms / 1e6);
        cudaFree(s), cudaFree(t);
    }
    printf("| decodificação | bits/peso | µs por matriz %d×%d | GB/s lidos | relativo ao uniforme do mesmo k | relativo ao uniforme de 4 bits | confere |\n",
           LINHAS, COLS);
    printf("|---|---:|---:|---:|---:|---:|---|\n");
    double base4 = 0, baseK = 0;
    std::vector<float> hy(LINHAS);
    for (int K = 0; K <= 2; K++)
        for (int livro = 0; livro <= 1; livro++) {
            auto roda = [&](const Mat &m) {
                const dim3 g(LINHAS / 8), bl(256);
#define CHAMA(KK, LL) mv<KK, LL><<<g, bl>>>(m.q, m.p1, m.p2, m.sc, m.d, ax, axp, ad, as, y)
                if (K == 0 && livro) CHAMA(0, true);
                if (K == 0 && !livro) CHAMA(0, false);
                if (K == 1 && livro) CHAMA(1, true);
                if (K == 1 && !livro) CHAMA(1, false);
                if (K == 2 && livro) CHAMA(2, true);
                if (K == 2 && !livro) CHAMA(2, false);
            };
            roda(M[NMAT - 1]);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hy.data(), y, LINHAS * 4, cudaMemcpyDeviceToHost));
            double erro = 0;
            for (int r = 0; r < 64; r++) {
                const double e = ref(K, livro, r);
                erro = std::max(erro, std::fabs(hy[r] - e) / (std::fabs(e) + 1e-3));
            }
            for (int r = 0; r < 3; r++)
                for (auto &m : M) roda(m);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0), cudaEventCreate(&e1);
            const int REP = 30;
            float melhor = 1e30f;
            for (int rep = 0; rep < 5; rep++) {
                cudaEventRecord(e0);
                for (int r = 0; r < REP; r++)
                    for (auto &m : M) roda(m);
                cudaEventRecord(e1);
                CK(cudaEventSynchronize(e1));
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                melhor = std::min(melhor, ms);
            }
            const double us = melhor * 1000.0 / REP / NMAT;
            const double bytes = nb * (16.0 + 4.0 * K + 1.0) + nb / 8 * 2.0;
            const double bpp = bytes * 8 / ((double)LINHAS * COLS);
            if (!livro) baseK = us;
            if (K == 0 && !livro) base4 = us;
            printf("| %s, %d bits | %.2f | %.1f | %.0f | %.2f× | %.2f× | %s |\n", livro ? "livro (tabela)" : "uniforme", 4 + K, bpp,
                   us, bytes / us / 1e3, baseK / us, base4 / us, erro < 1e-3 ? "ok" : "ERRO");
        }
    printf("\nrelativo = velocidade (tempo do outro / tempo deste): abaixo de 1 é mais lento.\n");
    return 0;
}
