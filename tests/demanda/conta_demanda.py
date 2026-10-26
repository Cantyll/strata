#!/usr/bin/env python3
"""Contas da precisão sob demanda (docs/08-precisao-sob-demanda.md), sem rodar modelo.

1. Linha do tempo de download do Llama-3.1-8B-Instruct: quando cada nível fica disponível começando pelo GGUF Q4_K_M
   do Hub e baixando os arquivos de estrato da seção 31, contra baixar o k-quant inteiro.
2. Degraus de RAM do repositório de GGUF do Hub (bartowski, 24 arquivos) entre o Q4_K_M e o Q6_K_L, para ver quanto
   espaço sobra para um tamanho contínuo.
3. Teto da decodificação especulativa com o nível 0 rascunhando e o nível 2 conferindo, pela conta de Leviathan et al.
   (2023): tokens por rodada E = (1 − α^(γ+1)) / (1 − α), custo = γ·c + v.

    python tests/demanda/conta_demanda.py [--alfa 0.9] [--c 0.71]
"""
import argparse, json, os

AQUI = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.join(AQUI, "..", "dist", "resultados", "hub_bartowski_Meta-Llama-3.1-8B-Instruct-GGUF.json")
GB, GIB = 1e9, 2 ** 30

# Seção 31 de docs/05-resultados.md (bytes reais dos arquivos gravados)
Q4KM = 4_920_739_232
NIVEL = 1_003_749_392          # model-e01, e02, e03: 1 bit cru por peso cada
E00 = 265_496
FECHO = 3_243_137_938
# RAM dos tipos no llama.cpp, 8B (seções 24 e 30): GGUF carregado, GiB
RAM = {"nível 0 (Q4_K_M)": 4.583, "nível 1 (P/D, Q6_K_P1 compacto)": 5.396, "nível 2 (P/D)": 6.208}


def linha_do_tempo(mbits):
    bps = mbits * 1e6 / 8
    t = lambda b: b / bps
    q4 = t(Q4KM + E00)
    return {"Q4_K_M": q4, "nível 1": q4 + t(NIVEL), "nível 2": q4 + t(2 * NIVEL),
            "BF16 exato": q4 + t(3 * NIVEL + FECHO)}


def hub():
    arq = json.load(open(HUB))
    out = {}
    for x in arq:
        nome, tam = x["path"], x["size"]
        if nome.endswith(".gguf") and "Q4_0_" not in nome and "f32" not in nome:
            out[nome.replace("Meta-Llama-3.1-8B-Instruct-", "").replace(".gguf", "")] = tam
    return dict(sorted(out.items(), key=lambda x: x[1]))


def especulativa(alfa, c, gama, v):
    e = (1 - alfa ** (gama + 1)) / (1 - alfa)
    return e / (gama * c + v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alfa", type=float, nargs="*", default=[0.85, 0.90, 0.95])
    ap.add_argument("--c", type=float, nargs="*", default=[0.71, 0.80])
    ap.add_argument("--v", type=float, default=1.05, help="custo de conferir γ+1 tokens de uma vez, em tokens do nível 2")
    a = ap.parse_args()

    print("1. Download do 8B a partir do zero (s), arquivos da seção 31")
    print(f"{'Mbit/s':>7} | {'Q4_K_M':>7} {'nível 1':>8} {'nível 2':>8} {'BF16':>7} | {'Q5_K_M':>7} {'Q6_K':>7} {'BF16 cru':>8}")
    tam = hub()
    for m in (50, 100, 300, 1000):
        d = linha_do_tempo(m)
        bps = m * 1e6 / 8
        print(f"{m:>7} | {d['Q4_K_M']:>7.0f} {d['nível 1']:>8.0f} {d['nível 2']:>8.0f} {d['BF16 exato']:>7.0f} | "
              f"{tam['Q5_K_M'] / bps:>7.0f} {tam['Q6_K'] / bps:>7.0f} {16.06e9 / bps:>8.0f}")
    print(f"quem já tem o Q4_K_M: nível 1 = +{NIVEL / GB:.2f} GB, nível 2 = +{2 * NIVEL / GB:.2f} GB "
          f"(Q5_K_M {tam['Q5_K_M'] / GB:.2f} GB, Q6_K {tam['Q6_K'] / GB:.2f} GB, Q6_K_L {tam['Q6_K_L'] / GB:.2f} GB)")

    print("\n2. Degraus do Hub entre 4,5 e 7 GiB (arquivo GGUF ≈ RAM dos pesos)")
    prev = None
    for n, b in tam.items():
        g = b / GIB
        if 4.4 < g < 7.0:
            gap = "" if prev is None else f"  degrau +{g - prev:.2f} GiB"
            print(f"  {n:>7} {g:6.3f} GiB{gap}")
            prev = g
    print("  STRATA (seção 30): " + ", ".join(f"{k} {v:.3f}" for k, v in RAM.items()) +
          "; com profundidade por tensor, qualquer valor entre 4,58 e 6,21 GiB")

    print(f"\n3. Teto da especulativa (rascunho = nível 0, conferência = nível 2, v = {a.v})")
    print("   c = custo do token do nível 0 / custo do token do nível 2 (memória: 4,80/6,80 b = 0,71)")
    for c in a.c:
        for alfa in a.alfa:
            melhor = max((especulativa(alfa, c, g, a.v), g) for g in range(1, 9))
            print(f"   c={c:.2f} α={alfa:.2f}: {melhor[0]:.2f}× com γ={melhor[1]}")
    print(f"   limite com α = 1: 1/c = {1 / a.c[0]:.2f}×")


if __name__ == "__main__":
    main()
