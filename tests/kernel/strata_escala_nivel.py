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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--gguf", required=True, help="Q4_K_M de partida (o nível 0)")
    ap.add_argument("--imatrix")
    ap.add_argument("--modo", choices=["fixo", "escala", "copia"], required=True)
    ap.add_argument("--tipos", nargs="+", default=["Q4_K", "Q6_K"], help="tipos de matriz que ganham escala própria")
    ap.add_argument("--grupo", choices=["sub", "bloco"], default="sub", help="uma escala por sub-bloco ou por bloco de 256")
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
        alvo = len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and (ty in BYTES or args.modo == "copia" and ty in ("Q5_K",))
        alvos.append(alvo)
        for wr in writers:
            if alvo:
                rows, ncols = t.data.shape[0], t.data.shape[1] // gguf.GGML_QUANT_SIZES[t.tensor_type][1] * 256
                wr.add_tensor_info(t.name, (rows, ncols), np.dtype(np.float16), rows * ncols * 2, raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    for wr in writers:
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()

    n_w = {"Q4_K": 0, "Q6_K": 0}
    hist = {}
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
        _, a, _ = celula(torch.from_numpy(raw), ty)
        passo = a.numpy()                                       # |passo| por sub-bloco, (n, sb, 1)
        qw = imat.get(t.name)
        w = np.ones_like(x) if qw is None else np.broadcast_to(
            np.tile(qw.astype(np.float64), rows).reshape(x.shape), x.shape)
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
    for k in range(1, K + 1):
        por = {"sub": {"Q4_K": 32, "Q6_K": 16}, "bloco": {"Q4_K": 256, "Q6_K": 256}}[args.grupo]
        extra = sum(n_w[ty] * 4 / por[ty] for ty in args.tipos) / tot if args.modo == "escala" else 0
        print(f"{args.out[k - 1]}: nível {k}, modo {args.modo}, +{extra:.3f} bit/peso de escalas por nível nas matrizes das camadas")
    for (ty, k), h in sorted(hist.items()):
        print(f"  m escolhido, {ty} nível {k}: " + " ".join(f"{i}:{100 * c / h.sum():.0f}%" for i, c in enumerate(h) if c))


if __name__ == "__main__":
    main()
