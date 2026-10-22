#!/usr/bin/env python3
"""Níveis com escala própria (experimento, só qualidade): o nível 0 continua o Q4_K_M (com imatrix, se ele foi feito
assim) e cada nível k guarda, além de 1 bit por peso, uma escala de 4 bits por sub-bloco, escolhida com a imatrix.

Hoje o nível k divide cada célula ao meio de forma fixa: v_k = v_(k−1) ± passo/2^(k+1), com o sinal de x − v_(k−1). Aqui o
sinal é o mesmo, mas o tamanho do passo de cada sub-bloco é a_k = m · passo/2^(k+4), com m de 0 a 15 escolhido para
minimizar Σ imatrix·(x − v_k)²; m = 8 é o passo fixo de hoje, então a busca nunca piora o erro ponderado. Custo: 4 bits
por sub-bloco por nível (0,125 bit/peso nas matrizes Q4_K, sub-blocos de 32; 0,25 nas Q6_K, sub-blocos de 16).

Sem kernel: os pesos de cada nível vão para um GGUF em F16, que o llama-perplexity lê direto, para medir a KL.

    python tests/kernel/strata_escala_nivel.py --bf16 BF16.gguf --gguf Q4_K_M.gguf --imatrix imatrix.gguf \\
        --modo escala --tipos Q4_K Q6_K --out n1.gguf n2.gguf

--modo desloc (vertente 3 de docs/07, "sub-células não uniformes"): o nível 0 é o próprio --gguf, intacto, e no nível k
cada célula do nível k − 1 se divide em dois filhos com deslocamentos próprios, m₋ e m₊ (0 … 31, em unidades de
passo/2^(k+4); m = 8 é o meio da metade, o de hoje), um par por código do nível anterior em cada tensor (--grupo tensor)
ou em cada linha (--grupo linha). Os pares saem de 3 rodadas de Lloyd ponderado pela imatrix: cada peso vai para o
filho mais perto, e cada deslocamento vira a média ponderada dos erros dos pesos do filho. O código do nível k é
2·(código do nível k − 1) + filho, no sentido de q crescente, como os planos do arquivo.

--modo fixo emula o STRATA de hoje (m = 8 sempre); os pesos saem iguais bit a bit aos dos tipos P (conferido em matrizes
Q4_K e Q6_K do Llama-3.2-1B). --modo copia grava só o próprio --gguf com as matrizes em F16 (um --out), para comparar
k-quants no mesmo caminho: em F16 o llama.cpp não arredonda as ativações para 8 bits, e a KL de todos cai.
"""
import argparse, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
sys.path.insert(0, os.path.join(ROOT, "tests", "kernel"))
from strata_base_fina import ler_imatrix  # noqa: E402
from strata_gguf_niveis import celula  # noqa: E402
from strata_ppl import GGUF_NAMES  # noqa: E402
from test_q4kr import BYTES  # noqa: E402


def niveis(x, v0, passo, w, K, modo, grupo="sub", m_fixo=None):
    """x, v0: (n, sub-blocos, pesos por sub-bloco) float64; passo: (n, sub-blocos, 1) > 0; w: importância, mesma forma
    de x. Devolve [v1, …, vK] e os m escolhidos por nível."""
    v, out, ms = v0, [], []
    s0 = w.sum(axis=2, keepdims=True)
    for k in range(1, K + 1):
        e = x - v
        s = np.where(e >= 0, 1.0, -1.0)
        un = passo / 2 ** (k + 4)
        if modo == "fixo":
            m = np.full(passo.shape, 8.0 if m_fixo is None else m_fixo[k - 1])
        elif grupo == "bloco":                                  # um m por bloco de 256 pesos
            s1 = (w * np.abs(e) * un).sum(axis=(1, 2), keepdims=True)
            s2 = (w * un * un).sum(axis=(1, 2), keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(s2 > 0, np.clip(np.rint(s1 / s2), 0, 15), 8.0) * np.ones(passo.shape)
        else:                                                   # erro ponderado é quadrático em m: arredonda o ótimo
            s1 = (w * np.abs(e)).sum(axis=2, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(un * s0 > 0, np.clip(np.rint(s1 / (un * s0)), 0, 15), 8.0)
        v = v + s * m * un
        out.append(v)
        ms.append(m)
    return out, ms


def niveis_desloc(x, v0, passo, cod, w, K, grupos_linha, iters=3, por="codigo", ncod0=None):
    """x, v0, w: (blocos, sub-blocos, pesos) float64; passo: (blocos, sub-blocos, 1) com sinal; cod: código do nível 0 de
    cada peso; grupos_linha: None (um par de deslocamentos por código no tensor) ou o número de blocos por linha (um par
    por código em cada linha). por: "codigo" (um par por código do nível anterior), "par" (um par só por nível, sem olhar
    o código) ou "borda" (o código do nível 0 conta só como mínimo, máximo ou meio; os bits dos níveis acima contam
    inteiros); ncod0 é o número de códigos do nível 0 (16 no Q4_K, 64 no Q6_K), para "borda" saber qual é o maior.
    Devolve [v1, …, vK] e o número de pares de cada nível."""
    sg = np.sign(passo)
    sg[sg == 0] = 1.0
    linha = None
    if grupos_linha:
        linha = (np.arange(x.shape[0]) // grupos_linha).reshape(-1, 1, 1) * np.ones(x.shape, np.int64)
    xs, vs, ws, c = x.ravel(), v0.ravel(), w.ravel(), cod.ravel().astype(np.int64)
    sgs = np.broadcast_to(sg, x.shape).ravel()
    ncod = int(c.max()) + 1
    if ncod0 is None:
        ncod0 = ncod
    out, pares = [], []
    for k in range(1, K + 1):
        un = np.broadcast_to(np.abs(passo) / 2 ** (k + 4), x.shape).ravel()
        e = (xs - vs) * sgs                                     # erro no sentido de q crescente
        if por == "par":
            gc, ng = np.zeros_like(c), 1
        elif por == "borda":
            c0, sub = c >> (k - 1), c & ((1 << (k - 1)) - 1)
            gc, ng = (np.where(c0 == 0, 0, np.where(c0 == ncod0 - 1, 2, 1)) << (k - 1)) | sub, 3 << (k - 1)
        else:
            gc, ng = c, ncod
        g = gc if linha is None else linha.ravel() * ng + gc
        G = int(g.max()) + 1
        b = e >= 0
        wu, wu2 = ws * un, ws * un * un
        for _ in range(iters):
            mp = np.bincount(g, wu * e * b, G) / np.maximum(np.bincount(g, wu2 * b, G), 1e-300)
            mm = -np.bincount(g, wu * e * ~b, G) / np.maximum(np.bincount(g, wu2 * ~b, G), 1e-300)
            mp = np.where(np.bincount(g, b.astype(np.float64), G) > 0, np.clip(np.rint(mp), 0, 31), 8.0)
            mm = np.where(np.bincount(g, (~b).astype(np.float64), G) > 0, np.clip(np.rint(mm), 0, 31), 8.0)
            b = np.abs(e - mp[g] * un) < np.abs(e + mm[g] * un)
        vs = vs + sgs * np.where(b, mp[g], -mm[g]) * un
        c = 2 * c + b
        ncod *= 2
        out.append(vs.reshape(x.shape))
        pares.append(G)
    return out, pares


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--gguf", required=True, help="Q4_K_M de partida (o nível 0)")
    ap.add_argument("--imatrix")
    ap.add_argument("--modo", choices=["fixo", "escala", "copia", "desloc"], required=True)
    ap.add_argument("--tipos", nargs="+", default=["Q4_K", "Q6_K"], help="tipos de matriz que ganham escala própria")
    ap.add_argument("--grupo", choices=["sub", "bloco", "tensor", "linha"], default="sub",
                    help="modo escala: uma escala por sub-bloco ou por bloco de 256; modo desloc: pares por tensor ou por linha")
    ap.add_argument("--por", choices=["codigo", "par", "borda"], default="codigo",
                    help="modo desloc: um par por código do nível anterior, um par só, ou só bordas/meio do nível 0")
    ap.add_argument("--m", type=float, nargs="+", help="no modo fixo, o m de cada nível (padrão 8, o passo de hoje)")
    ap.add_argument("--out", nargs="+", required=True, help="um GGUF F16 por nível, 1 … K")
    args = ap.parse_args()
    K = 0 if args.modo == "copia" else len(args.out)

    import gguf
    imat = ler_imatrix(args.imatrix, gguf) if args.imatrix else {}
    bf = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    reader = gguf.GGUFReader(args.gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writers = []
    for out in args.out:
        wr = gguf.GGUFWriter(out, arch)
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            wr.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
        writers.append(wr)
    alvos = []
    for t in reader.tensors:
        p = t.name.split(".")
        ty = t.tensor_type.name
        alvo = len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and (ty in BYTES or args.modo == "copia" and ty in ("Q5_K", "Q8_0"))
        alvos.append(alvo)
        for wr in writers:
            if alvo:
                rows, ncols = t.data.shape[0], t.data.shape[1] // gguf.GGML_QUANT_SIZES[t.tensor_type][1] * gguf.GGML_QUANT_SIZES[t.tensor_type][0]
                wr.add_tensor_info(t.name, (rows, ncols), np.dtype(np.float16), rows * ncols * 2, raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    for wr in writers:
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()

    n_w = {"Q4_K": 0, "Q6_K": 0}
    hist = {}
    bits_tab = np.zeros(K + 1)                                  # bits das tabelas de deslocamento (modo desloc), por nível
    for t, alvo in zip(reader.tensors, alvos):
        data = np.array(t.data)
        if not alvo:
            for wr in writers:
                wr.write_tensor_data(data)
            continue
        ty = t.tensor_type.name
        rows = data.shape[0]
        if args.modo == "copia":
            writers[0].write_tensor_data(gguf.quants.dequantize(data, t.tensor_type).astype(np.float16))
            continue
        raw = data.reshape(-1, BYTES[ty])
        u16 = np.array(bf[t.name].data).view(np.uint16).reshape(rows, -1)
        ncols = u16.shape[1]
        n_w[ty] += rows * ncols
        sb = 8 if ty == "Q4_K" else 16
        x = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(-1, sb, 256 // sb)
        v0 = gguf.quants.dequantize(data, t.tensor_type).astype(np.float64).reshape(x.shape)
        _, a, ps, cod = celula(torch.from_numpy(raw), ty, codigo=True)
        passo = a.numpy()                                       # |passo| por sub-bloco, (n, sb, 1)
        qw = imat.get(t.name)
        w = np.ones_like(x) if qw is None else np.broadcast_to(
            np.tile(qw.astype(np.float64), rows).reshape(x.shape), x.shape)
        if args.modo == "desloc":
            vs, pares = niveis_desloc(x, v0, ps.numpy(), cod.numpy(), w, K,
                                      ncols // 256 if args.grupo == "linha" else None, por=args.por,
                                      ncod0=16 if ty == "Q4_K" else 64)
            for k, (wr, v) in enumerate(zip(writers, vs)):
                wr.write_tensor_data(v.reshape(rows, ncols).astype(np.float16))
                bits_tab[k + 1] += 2 * pares[k] * 5
            continue
        vs, ms = niveis(x, v0, passo, w, K, args.modo if ty in args.tipos else "fixo", args.grupo,
                        args.m if args.modo == "fixo" or ty in args.tipos else None)
        for k, (wr, v) in enumerate(zip(writers, vs)):
            wr.write_tensor_data(v.reshape(rows, ncols).astype(np.float16))
            if ty in args.tipos and args.modo == "escala":
                hist.setdefault((ty, k + 1), np.zeros(16, np.int64))
                hist[(ty, k + 1)] += np.bincount(ms[k].astype(np.int64).ravel(), minlength=16)
    for wr in writers:
        wr.close()
    if args.modo == "copia":
        print(f"{args.out[0]}: matrizes de {args.gguf} em F16")
        return
    tot = sum(n_w.values())
    if args.modo == "desloc":
        for k in range(1, K + 1):
            print(f"{args.out[k - 1]}: nível {k}, deslocamentos por {args.grupo}, tabelas de {bits_tab[k] / 8 / 2**20:.2f} MiB "
                  f"(+{bits_tab[k] / tot:.4f} bit/peso nas matrizes das camadas)")
        return
    for k in range(1, K + 1):
        por = {"sub": {"Q4_K": 32, "Q6_K": 16}, "bloco": {"Q4_K": 256, "Q6_K": 256}}[args.grupo]
        extra = sum(n_w[ty] * 4 / por[ty] for ty in args.tipos) / tot if args.modo == "escala" else 0
        print(f"{args.out[k - 1]}: nível {k}, modo {args.modo}, +{extra:.3f} bit/peso de escalas por nível nas matrizes das camadas")
    for (ty, k), h in sorted(hist.items()):
        print(f"  m escolhido, {ty} nível {k}: " + " ".join(f"{i}:{100 * c / h.sum():.0f}%" for i, c in enumerate(h) if c))


if __name__ == "__main__":
    main()
