"""Régua do C1 + C2 no Llama-3.2-3B (docs/10, "O 3B no contêiner"), sobre os JSONs de strata_dkl.py teste/avalia.

    python regua_3b.py /home/user/work/b3/res /home/user/work/b3/Q4_K_M.gguf > regua.md

RAM = bytes de todos os tensores (como no 1B). Nível k = Q4_K_M + k bits por peso nas matrizes das camadas; mistura =
Q4_K_M + os bytes que o plano soma. "Bits a mais" = quanto gasta além do k-quant de mesma KL (log da KL interpolado
entre os k-quants vizinhos da frente). "Níveis inteiros" = log da KL interpolado entre os níveis inteiros vizinhos.
"""
import glob, json, math, os, sys

import gguf

R, Q4F = sys.argv[1], os.path.getsize(sys.argv[2])
TX = {"wiki": "wiki.test.raw", "c4": "c4.test.raw"}
KQ = ["Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]


def le(n):
    f = f"{R}/{n}.json"
    return json.load(open(f)) if os.path.exists(f) else None


tam = {q: int(open(f"{R}/tam_{q}.txt").read()) / 2**20 for q in KQ if os.path.exists(f"{R}/tam_{q}.txt")}
kl = {}
for n in KQ + ["N1", "N2"] + [os.path.basename(f)[:-5] for f in glob.glob(f"{R}/dkl_*.json") + glob.glob(f"{R}/mse_*.json")]:
    j = le(n)
    if j:
        for t, arq in TX.items():
            kl[(n, t)] = (j[arq]["kl"], j[arq]["erro"], j[arq]["top1"])

r = gguf.GGUFReader(sys.argv[2])
NCAM = sum(int(t.n_elements) for t in r.tensors if t.name.startswith("blk.") and len(t.shape) == 2)
Q4 = tam["Q4_K_M"]
RAM = {0: Q4, 1: Q4 + NCAM / 8 / 2**20, 2: Q4 + 2 * NCAM / 8 / 2**20}
NOME = {0: "Q4_K_M", 1: "N1", 2: "N2"}


def frente(t):
    pts = sorted((tam[q], kl[(q, t)][0]) for q in KQ if (q, t) in kl)
    return [p for p in pts if not any(o[0] <= p[0] and o[1] < p[1] for o in pts if o != p)]


def tam_igual(k, t):
    pts = frente(t)
    for (s1, k1), (s2, k2) in zip(pts, pts[1:]):
        if k >= k2 or (s2, k2) == pts[-1]:
            if k > pts[0][1]:
                (s1, k1), (s2, k2) = pts[0], pts[1]
            return s1 + (s2 - s1) * math.log(k / k1) / math.log(k2 / k1)


def bits(ram, k, t):
    return (ram - tam_igual(k, t)) * 2**20 * 8 / NCAM


def inteiros(ram, t):
    k = 0 if ram < RAM[1] else 1
    a = (ram - RAM[k]) / (RAM[k + 1] - RAM[k])
    l0, l1 = (math.log(kl[(NOME[j], t)][0]) for j in (k, k + 1))
    return math.exp(l0 + a * (l1 - l0))


L = [f"Matrizes das camadas: {NCAM / 1e9:.3f} bilhões de pesos. RAM do nível 0 (Q4_K_M) {RAM[0]:.1f} MiB, nível 1 {RAM[1]:.1f}, nível 2 {RAM[2]:.1f}.",
     "", "## Frente e níveis inteiros (KL ×1000, PyTorch, 40 trechos)", "",
     "| modelo | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais |", "|---|---:|---:|---:|---:|---:|"]
out = {"ncam": NCAM, "ram_niveis": RAM, "tam": tam, "kl": {f"{n}|{t}": v for (n, t), v in kl.items()}}
for n in KQ + ["N1", "N2"]:
    if (n, "wiki") not in kl:
        continue
    s = tam.get(n) or RAM[int(n[1])]
    w, c = kl[(n, "wiki")], kl[(n, "c4")]
    bw, bc = (f"{bits(s, w[0], 'wiki'):+.3f}", f"{bits(s, c[0], 'c4'):+.3f}") if n[0] == "N" else ("", "")
    L.append(f"| {n.replace('N', 'nível ')} | {s:.1f} | {1000 * w[0]:.2f} ± {1000 * w[1]:.2f} | {bw} | {1000 * c[0]:.2f} ± {1000 * c[1]:.2f} | {bc} |")

orc = sorted({f.split("_")[-1][:-5] for f in glob.glob(f"{R}/p_dkl_*.json")}, key=float)
res = {}
if orc and ("N2", "wiki") in kl:
    L += ["", "## Misturas (KL ×1000)", "",
          "| orçamento (GiB) | alocador | RAM (MiB) | wikitext | bits a mais | C4 | bits a mais | níveis inteiros (wiki / C4) | bits dos inteiros |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for o in orc:
        for a in ("dkl", "mse"):
            n = f"{a}_{o}"
            if (n, "wiki") not in kl:
                continue
            ram = Q4 + (json.load(open(f"{R}/p_{n}.json"))["gib"] * 2**30 - Q4F) / 2**20
            w, c = kl[(n, "wiki")], kl[(n, "c4")]
            iw, ic = inteiros(ram, "wiki"), inteiros(ram, "c4")
            v = res[n] = dict(ram=ram, kl=(w[0], c[0]), err=(w[1], c[1]), bits=(bits(ram, w[0], "wiki"), bits(ram, c[0], "c4")),
                              inteiros=(iw, ic), bits_inteiros=(bits(ram, iw, "wiki"), bits(ram, ic, "c4")))
            L.append(f"| {o} | {'medido' if a == 'dkl' else 'de hoje'} | {ram:.1f} | {1000 * w[0]:.2f} ± {1000 * w[1]:.2f} | {v['bits'][0]:+.3f} | "
                     f"{1000 * c[0]:.2f} ± {1000 * c[1]:.2f} | {v['bits'][1]:+.3f} | {1000 * iw:.2f} / {1000 * ic:.2f} | "
                     f"{v['bits_inteiros'][0]:+.3f} / {v['bits_inteiros'][1]:+.3f} |")
    out["misturas"] = res
    av = json.load(open(f"{R}/avalia.json")) if os.path.exists(f"{R}/avalia.json") else {}
    erro = {}
    if av:
        L += ["", "## A soma prevê a medida? (textos da medida)", "", "| orçamento | prevista | medida | erro |", "|---|---:|---:|---:|"]
        for o in orc:
            p = json.load(open(f"{R}/p_dkl_{o}.json"))["kl_prevista"]
            m = av[f"{R}/p_dkl_{o}.json"]
            erro[o] = p / m - 1
            L.append(f"| {o} | {1000 * p:.3f} | {1000 * m:.3f} | {100 * erro[o]:+.1f}% |")
        out["erro_soma"] = erro
    if len(res) == 2 * len(orc) and erro:
        ganha = [all(res[f"dkl_{o}"]["kl"][i] < res[f"mse_{o}"]["kl"][i] for i in (0, 1)) for o in orc]
        nao_pior = all(res[f"dkl_{o}"]["kl"][i] <= res[f"mse_{o}"]["kl"][i] + math.hypot(res[f"dkl_{o}"]["err"][i], res[f"mse_{o}"]["err"][i])
                       for o in orc for i in (0, 1))
        dentro = all(res[f"dkl_{o}"]["bits"][i] <= res[f"dkl_{o}"]["bits_inteiros"][i] + 0.10 for o in orc for i in (0, 1))
        soma = all(abs(e) <= 0.10 for e in erro.values())
        if sum(ganha) < 2:
            V = "FRACASSO: o medido não ganha do alocador de hoje em pelo menos dois orçamentos."
        elif dentro and nao_pior and soma:
            V = "SUCESSO: ≤ +0,10 bit além dos níveis inteiros nos dois textos, nenhum orçamento pior que o de hoje e a soma prevista erra ≤ 10%."
        else:
            V = "ZONA CINZA: ganha do de hoje, mas " + ", ".join(x for x, ok in (
                ("passa de +0,10 bit além dos níveis inteiros", dentro), ("algum orçamento fica pior que o de hoje além de 1 barra", nao_pior),
                ("a soma prevista erra mais de 10%", soma)) if not ok) + "."
        L += ["", f"**Veredito do C2 pela régua:** {V}", "", "Ganho do medido sobre os níveis inteiros, em bits (wiki / C4): " + ", ".join(
            f"{o} GiB {res[f'dkl_{o}']['bits_inteiros'][0] - res[f'dkl_{o}']['bits'][0]:+.3f} / "
            f"{res[f'dkl_{o}']['bits_inteiros'][1] - res[f'dkl_{o}']['bits'][1]:+.3f}" for o in orc) + "."]
        out["veredito"] = V
if ("N1", "c4") in kl:
    b1 = bits(RAM[1], kl[("N1", "c4")][0], "c4")
    L += ["", f"**C1: o 3B repete o atraso do 8B?** Nível 1 no C4: {b1:+.3f} bit (regra: ≥ +0,08 repete o 8B; ≤ +0,04 promete demais como o 1B)."]
print("\n".join(L))
json.dump(out, open(f"{R}/regua.json", "w"), indent=1)
