#!/usr/bin/env python3
"""E6 da precisão sob demanda (docs/08): o índice que o `strata_subir` usa para subir de nível lendo os planos direto
dos arquivos de estrato da distribuição B.10 (`strata-ref distribui`), em vez do arquivo de planos do E3.

O `model.strata.index.json` fala em nomes do HF; o `strata_subir` precisa dos nomes do GGUF. A ligação é o stream `q`
de cada matriz, que aponta para dentro do GGUF: o offset dele é o offset dos dados do tensor no GGUF. Saem só as
matrizes das camadas (blk.*), as que ganham níveis no E2.

    python tests/demanda/estratos_idx.py PASTA/model.strata.index.json Q4_K_M.gguf PASTA/subida.idx [--niveis 2]

Formato (texto): a primeira linha é `#arquivos e01 e02` (nomes relativos à pasta do .idx), e cada linha seguinte é
`nome grade nblocos off1 off2 sha1 sha2`, com o offset de cada plano no seu arquivo e o SHA-256 do stream, que o
`strata_subir` confere antes de usar o plano. As linhas vêm na ordem dos planos no arquivo (a do HF, não a do GGUF),
que é a ordem em que chegam pela rede.
"""
import argparse, json, os, sys

import gguf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indice")
    ap.add_argument("gguf")
    ap.add_argument("saida")
    ap.add_argument("--niveis", type=int, default=2)
    a = ap.parse_args()
    ix = json.load(open(a.indice))
    if ix.get("desenho") != "por-estrato-gguf":
        sys.exit("o índice não é do desenho por-estrato-gguf (B.10)")
    arqs = ix["arquivos"]
    por_offset = {t.data_offset: t for t in gguf.GGUFReader(a.gguf).tensors}
    q, planos = {}, {}
    for s in ix["streams"]:
        if s["p"] == "q" and s["f"] == 0:
            q[s["t"]] = s["o"]
        elif s["p"] in {f"n{k}" for k in range(1, a.niveis + 1)}:
            if s["c"] != "raw":
                sys.exit(f"stream {s['p']} do tensor {s['t']} não é cru")
            planos[(s["t"], int(s["p"][1:]))] = s
    nomes = {}
    for k in range(1, a.niveis + 1):
        fs = {planos[c]["f"] for c in planos if c[1] == k}
        if len(fs) != 1:
            sys.exit(f"os planos do nível {k} não estão num arquivo só")
        nomes[k] = arqs[fs.pop()]["nome"]
    linhas = []
    for t, o in q.items():
        tg = por_offset.get(o)
        if tg is None:
            sys.exit(f"o stream q do tensor {ix['tensors'][t]['name']} não aponta para o começo de um tensor do GGUF")
        if not tg.name.startswith("blk."):
            continue
        grade = ix["tensors"][t]["residual"]["grade"]
        if grade != tg.tensor_type.name:
            sys.exit(f"{tg.name}: o índice diz {grade}, o GGUF diz {tg.tensor_type.name}")
        n = ix["tensors"][t]["n"]
        ps = [planos[(t, k)] for k in range(1, a.niveis + 1)]
        if any(p["l"] != n // 8 for p in ps):
            sys.exit(f"{tg.name}: plano com tamanho errado")
        linhas.append((ps[0]["o"], f"{tg.name} {grade} {n // 256} " + " ".join(str(p["o"]) for p in ps) + " " + " ".join(p["h"] for p in ps)))
    with open(a.saida, "w") as f:
        f.write("#arquivos " + " ".join(nomes[k] for k in range(1, a.niveis + 1)) + "\n")
        for _, l in sorted(linhas):
            f.write(l + "\n")
    print(f"{a.saida}: {len(linhas)} matrizes, planos em " + ", ".join(nomes.values()))


if __name__ == "__main__":
    main()
