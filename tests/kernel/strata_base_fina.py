#!/usr/bin/env python3
"""Base "fina" do modo residual (experimento): as matrizes Q4_K das camadas são quantizadas direto na grade de 4 + T bits
(com a imatrix, se houver) e o Q4_K sai por corte dos bits baixos (tests/kernel/strata_base_fina.c). Grava um GGUF por
nível, de 0 a T: o nível 0 é um GGUF Q4_K_M comum (roda no llama.cpp sem patch), os níveis 1 … T usam os tipos
STRATA_Q4_K_P<k>/STRATA_Q6_K_P<k> do patch de kernel/ggml, como strata_gguf_niveis.py.

As matrizes Q6_K do Q4_K_M de partida ficam como estão no nível 0 e recebem os níveis do jeito de sempre (célula do Q6_K
subdividida). Os pesos vêm do GGUF BF16, que já está na ordem do llama.cpp.

    python tests/kernel/strata_base_fina.py --bf16 BF16.gguf --gguf Q4_K_M.gguf [--imatrix imatrix.gguf] --T 2 \\
        --out n0.gguf n1.gguf n2.gguf
"""
import argparse, ctypes, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
sys.path.insert(0, os.path.join(ROOT, "tests", "kernel"))
from strata_gguf_niveis import LOTE, TIPOS, celula, planos  # noqa: E402
from strata_ppl import GGUF_NAMES  # noqa: E402
from test_q4kr import BYTES, build_lib  # noqa: E402


def lib_fina():
    src = os.path.join(ROOT, "tests", "kernel", "strata_base_fina.c")
    so = os.path.join(os.environ.get("TMPDIR", "/tmp"), "libstrata_base_fina.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O2", "-ffp-contract=off", "-shared", "-fPIC", src, "-lm", "-o", so])
    lib = ctypes.CDLL(so)
    lib.strata_base_fina_q4k_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                              ctypes.c_int64, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return lib


def ler_imatrix(path, gguf):
    """{nome do tensor: importância por coluna}, normalizada pelas contagens como o llama-quantize faz."""
    r = gguf.GGUFReader(path)
    t = {x.name: np.array(x.data, dtype=np.float32).reshape(-1) for x in r.tensors}
    out = {}
    for nome, s in t.items():
        if nome.endswith(".in_sum2"):
            base = nome[: -len(".in_sum2")]
            c = t[base + ".counts"]
            out[base] = s / c[0] if c[0] > 0 else np.ones_like(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--gguf", required=True, help="Q4_K_M de partida (tipos por tensor, metadados, tensores que não mudam)")
    ap.add_argument("--imatrix")
    ap.add_argument("--T", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--pesos", type=float, nargs="+", help="peso do erro de cada nível 0 … T na busca das escalas "
                    "(padrão: só o nível T)")
    ap.add_argument("--out", nargs="+", required=True, help="um GGUF por nível, 0 … T")
    args = ap.parse_args()
    if len(args.out) != args.T + 1:
        raise SystemExit("--out precisa de T + 1 arquivos")

    import gguf
    for (_, _), (tid, nbytes) in TIPOS.items():
        gguf.constants.GGML_QUANT_SIZES[tid] = (256, nbytes)
    lib, fina = build_lib(), lib_fina()
    imat = ler_imatrix(args.imatrix, gguf) if args.imatrix else {}
    bf = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    _, vals, _, _ = __import__("strata_residual").bf16_table()
    vals = torch.from_numpy(vals)

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
        alvo = len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and ty in BYTES
        alvos.append(alvo)
        for k, wr in enumerate(writers):
            if alvo and k > 0:
                tid, pb = TIPOS[(ty, k)]
                rows, nb = t.data.shape[0], t.data.shape[1] // BYTES[ty]
                wr.add_tensor_info(t.name, (rows, nb * pb), np.dtype(np.uint8), rows * nb * pb, raw_dtype=tid)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    for wr in writers:
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()

    T = args.T
    NT = os.cpu_count() or 1
    pool = ThreadPoolExecutor(NT)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vals = vals.to(dev)
    pesos = None
    if args.pesos:
        if len(args.pesos) != T + 1:
            raise SystemExit("--pesos precisa de T + 1 números")
        pesos = np.array(args.pesos, dtype=np.float32)
    for t, alvo in zip(reader.tensors, alvos):
        data = np.array(t.data)
        if not alvo:
            for wr in writers:
                wr.write_tensor_data(data)
            continue
        ty = t.tensor_type.name
        rows = data.shape[0]
        u16 = np.ascontiguousarray(np.array(bf[t.name].data).view(np.uint16).reshape(rows, -1))
        ncols = u16.shape[1]
        if ty == "Q4_K":
            x = (u16.astype(np.uint32) << 16).view(np.float32)
            qw = imat.get(t.name)
            raw = np.empty((rows, ncols // 256 * 144), dtype=np.uint8)
            u = np.empty((rows, ncols), dtype=np.uint8)
            qw = None if qw is None else np.ascontiguousarray(qw, dtype=np.float32)
            por = -(-rows // NT)                                # linhas por thread (o ctypes solta o GIL)
            def faixa(r0):
                n = min(por, rows - r0)
                fina.strata_base_fina_q4k_rows(x[r0:].ctypes.data, raw[r0:].ctypes.data, u[r0:].ctypes.data, n, ncols,
                                               None if qw is None else qw.ctypes.data, T,
                                               None if pesos is None else pesos.ctypes.data)
            list(pool.map(faixa, range(0, rows, por)))
            raw = raw.reshape(-1, 144)
            u = u.reshape(-1, 256)
            baixos = u & ((1 << T) - 1)
            for k, wr in enumerate(writers):
                if k == 0:
                    wr.write_tensor_data(raw.reshape(rows, -1))
                    continue
                idx = torch.from_numpy((baixos >> (T - k)).astype(np.int64)).reshape(-1, 32, 8)
                peso = (1 << torch.arange(8)).reshape(1, 1, 8)
                pl = torch.cat([(((idx >> (k - p)) & 1) * peso).sum(-1).to(torch.uint8) for p in range(1, k + 1)], dim=1).numpy()
                blocks = np.ascontiguousarray(np.concatenate([raw, pl], axis=1))
                tid, pb = TIPOS[(ty, k)]
                dst = np.empty((rows, ncols // 256 * pb), dtype=np.uint8)
                lib.strata_q4kr_to_q4kp(k, blocks.ctypes.data, dst.ctypes.data, rows * ncols)
                wr.write_tensor_data(dst)
        else:                                                   # Q6_K: célula do bloco subdividida, como antes
            raw = data.reshape(-1, BYTES[ty])
            uu = u16.reshape(-1, 256)
            planes = {k: [] for k in range(1, T + 1)}
            for i in range(0, raw.shape[0], LOTE):
                lo, a, passo = celula(torch.from_numpy(raw[i:i + LOTE]).to(dev), ty)
                w = vals[torch.from_numpy(uu[i:i + LOTE].astype(np.int64)).to(dev)].reshape(lo.shape)
                for k in planes:
                    planes[k].append(planos(w, lo, a, passo, k).cpu().numpy())
            for k, wr in enumerate(writers):
                if k == 0:
                    wr.write_tensor_data(data)
                    continue
                blocks = np.ascontiguousarray(np.concatenate([raw, np.concatenate(planes[k])], axis=1))
                tid, pb = TIPOS[(ty, k)]
                dst = np.empty((rows, ncols // 256 * pb), dtype=np.uint8)
                lib.strata_q6kr_to_q6kp(k, blocks.ctypes.data, dst.ctypes.data, rows * ncols)
                wr.write_tensor_data(dst)
    for wr in writers:
        wr.close()
    for k, out in enumerate(args.out):
        print(f"{out}: nível {k} da base fina T={T}{f' pesos {args.pesos}' if pesos is not None else ''}{' com imatrix' if imat else ''}, {os.path.getsize(out) / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
