#!/usr/bin/env python3
"""Confere a referência em C dos tipos Q4_K_R<k> e Q6_K_R<k> (kernel/strata_q4kr.c) num modelo real.

Para cada matriz Q4_K ou Q6_K de um GGUF Q4_K_M:
  1. monta os blocos Q4_K_R<k> (ou Q6_K_R<k>): o bloco do GGUF byte a byte + k planos com os bits dos níveis 1..k de cada peso,
     tirados dos pesos BF16 como em strata_residual.py (saturados cortados na sub-célula da ponta);
  2. desquantiza com o C e confere bit a bit contra a mesma fórmula em float32 no numpy;
  3. compara com a emulação em float64 de strata_residual.reconstruct (a que deu os ΔPPL do Colab);
  4. confere o produto escalar inteiro (contra ativações Q8_K) contra o produto em float64.
Compila o C com -ffp-contract=off. Imprime um resumo por k e sai com código 1 se algo divergir.

    python tests/kernel/test_q4kr.py --model PASTA_OU_ID --gguf Q4_K_M.gguf [--levels 1,2,3] [--limit N] [--tipos Q4_K,Q6_K]
"""
import argparse, ctypes, os, subprocess, sys, tempfile, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
from strata_ppl import GGUF_NAMES  # noqa: E402
from strata_residual import bf16_table, grid, nominal_cell, reconstruct, sub_index  # noqa: E402
from strata_to_gguf import permute_like_gguf  # noqa: E402

Q8_BYTES = 4 + 256 + 32
HAS_AVX2 = "avx2" in open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else False


def build_lib():
    out = os.path.join(tempfile.mkdtemp(), "libq4kr.so")
    src = [os.path.join(ROOT, "kernel", "strata_q4kr.c")]
    flags = []
    if HAS_AVX2:
        src.append(os.path.join(ROOT, "kernel", "strata_q4kr_avx2.c"))
        flags = ["-mavx2", "-mfma", "-mf16c"]
    subprocess.run(["cc", "-std=c99", "-O2", "-ffp-contract=off", *flags, "-fPIC", "-shared", "-o", out, *src, "-lm"], check=True)
    lib = ctypes.CDLL(out)
    p = ctypes.c_void_p
    lib.strata_q4kr_dequantize_row.argtypes = [ctypes.c_int, p, p, ctypes.c_int64]
    lib.strata_q8_K_quantize_row.argtypes = [p, p, ctypes.c_int64]
    lib.strata_q4kr_vec_dot_q8_K.argtypes = [ctypes.c_int, p, p, ctypes.c_int64]
    lib.strata_q4kr_vec_dot_q8_K.restype = ctypes.c_float
    lib.strata_q6kr_dequantize_row.argtypes = [ctypes.c_int, p, p, ctypes.c_int64]
    lib.strata_q6kr_vec_dot_q8_K.argtypes = [ctypes.c_int, p, p, ctypes.c_int64]
    lib.strata_q6kr_vec_dot_q8_K.restype = ctypes.c_float
    if HAS_AVX2:
        for f in (lib.strata_q4kr_vec_dot_q8_K_avx2, lib.strata_q6kr_vec_dot_q8_K_avx2):
            f.argtypes = [ctypes.c_int, p, p, ctypes.c_int64]
            f.restype = ctypes.c_float
    return lib


def pack(raw, idx, k):
    """raw: blocos do GGUF (n, 144 ou 210); idx: sub-célula de cada peso (n, 256), 0..2^k-1. Devolve (n, bytes + 32k)."""
    planes = [np.packbits(((idx >> (k - p)) & 1).astype(np.uint8), axis=1, bitorder="little") for p in range(1, k + 1)]
    return np.ascontiguousarray(np.concatenate([raw] + planes, axis=1))


def dequant_f32(raw, idx, k, gguf):
    """A fórmula do header em float32, operação por operação (o que o C tem de dar bit a bit)."""
    n = raw.shape[0]
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n, 1)
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n, 1)
    sc, mn = gguf.quants.Q4_K.get_scale_min(raw[:, 4:16].copy())
    qs = raw[:, 16:144].reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    q4 = (qs & 0x0F).reshape(n, 8, 32).astype(np.int64)
    a = (d * sc.astype(np.float32)).reshape(n, 8, 1)
    m = (dmin * mn.astype(np.float32)).reshape(n, 8, 1)
    t = (q4 << (k + 1)) + 2 * idx.reshape(n, 8, 32) + 1 - (1 << k)
    inv = np.float32(1.0 / (1 << (k + 1)))
    return (a * (t.astype(np.float32) * inv) - m).astype(np.float32).reshape(-1)


def dequant_q6_f32(raw, q, idx, k):
    """Q6_K_R em float32: q (n, 256) de −32 a 31 vem da desquantização do gguf-py, não do parser do C."""
    n = raw.shape[0]
    d = raw[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(n, 1)
    sc = np.repeat(raw[:, 192:208].copy().view(np.int8).astype(np.float32), 16, axis=1)
    a = d * sc
    t = q * (1 << (k + 1)) + 2 * idx + 1 - (1 << k)
    inv = np.float32(1.0 / (1 << (k + 1)))
    return (a * (t.astype(np.float32) * inv)).astype(np.float32).reshape(-1)


BYTES = {"Q4_K": 144, "Q6_K": 210}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--gguf", required=True, help="GGUF Q4_K_M (estrato 0)")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--limit", type=int, default=0, help="só as primeiras N matrizes")
    ap.add_argument("--tipos", default="Q4_K,Q6_K", help="tipos do GGUF a conferir")
    ap.add_argument("--dot-rows", type=int, default=8, help="linhas por matriz no teste do produto escalar")
    args = ap.parse_args()

    import gguf
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    lib = build_lib()
    path = args.model if os.path.isdir(args.model) else snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(path)
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads or cfg.num_attention_heads
    sd = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(path, f)))
    _, vals, _, _ = bf16_table()
    levels = [int(x) for x in args.levels.split(",")]
    rng = np.random.default_rng(0)
    st = {(ty, k): dict(n=0, m=0, bad=0, avx2_bad=0, emu=0.0, dot=0.0, deq_s=0.0, dot_s=0.0, rows=0) for ty in BYTES for k in levels}
    done = 0
    for t in gguf.GGUFReader(args.gguf).tensors:
        p = t.name.split(".")
        ty = t.tensor_type.name
        if not (len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and ty in BYTES and ty in args.tipos.split(",")):
            continue
        nb = BYTES[ty]
        deq_c = lib.strata_q4kr_dequantize_row if ty == "Q4_K" else lib.strata_q6kr_dequantize_row
        dot_c = lib.strata_q4kr_vec_dot_q8_K if ty == "Q4_K" else lib.strata_q6kr_vec_dot_q8_K
        dot_v = (lib.strata_q4kr_vec_dot_q8_K_avx2 if ty == "Q4_K" else lib.strata_q6kr_vec_dot_q8_K_avx2) if HAS_AVX2 else None
        name = f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}"
        wt = sd[name + ".weight"]
        u = wt.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        w = permute_like_gguf(vals[u], name, tuple(wt.shape), n_head, n_kv)
        data = np.array(t.data)
        rows, ncols = data.shape[0], data.shape[1] // nb * 256
        raw = data.reshape(-1, nb)
        y, step, q, qmin, qmax, _ = grid(t, gguf)
        lo, a, _, _, _ = nominal_cell(y, step, q, qmin, qmax, w)
        x = rng.standard_normal(ncols).astype(np.float32)
        q8 = np.zeros(ncols // 256 * Q8_BYTES, dtype=np.uint8)
        lib.strata_q8_K_quantize_row(x.ctypes.data, q8.ctypes.data, ncols)
        qb = q8.reshape(-1, Q8_BYTES)
        xq = (qb[:, 4:260].view(np.int8).astype(np.float64) * qb[:, 0:4].copy().view(np.float32).astype(np.float64)).reshape(-1)
        for k in levels:
            idx = sub_index(np.clip(w, lo, lo + a), lo, a, k).astype(np.int64)
            # sub_index conta no sentido do valor; o tipo conta no sentido de q (difere onde o passo é negativo)
            idx = np.where(step < 0, (1 << k) - 1 - idx, idx).reshape(-1, 256)
            blocks = pack(raw, idx, k)
            out = np.empty(w.size, dtype=np.float32)
            t0 = time.time()
            deq_c(k, blocks.ctypes.data, out.ctypes.data, w.size)
            s_ = st[(ty, k)]
            s_["deq_s"] += time.time() - t0
            if ty == "Q4_K":
                ref = dequant_f32(raw, idx, k, gguf)
            else:
                ref = dequant_q6_f32(raw, q.astype(np.int64).reshape(-1, 256), idx, k)
            s_["bad"] += int((out.view(np.uint32) != ref.view(np.uint32)).sum())
            emu = reconstruct(w, y, step, q, qmin, qmax, k, exact_out=False)
            s_["emu"] = max(s_["emu"], float(np.max(np.abs(out - emu)) / np.max(np.abs(a))))
            br = blocks.reshape(rows, -1)
            t0 = time.time()
            for r in range(min(args.dot_rows, rows)):
                got = dot_c(k, br[r].ctypes.data, q8.ctypes.data, ncols)
                if dot_v is not None:
                    gv = dot_v(k, br[r].ctypes.data, q8.ctypes.data, ncols)
                    s_["avx2_bad"] += int(np.float32(gv).view(np.uint32) != np.float32(got).view(np.uint32))
                want = float(np.dot(out[r * ncols:(r + 1) * ncols].astype(np.float64), xq))
                scale = float(np.sum(np.abs(out[r * ncols:(r + 1) * ncols].astype(np.float64) * xq))) or 1.0
                s_["dot"] = max(s_["dot"], abs(got - want) / scale)
                s_["rows"] += 1
            s_["dot_s"] += time.time() - t0
            s_["n"] += w.size
            s_["m"] += 1
        done += 1
        if args.limit and done >= args.limit:
            break
    fails = 0
    for (ty, k), s in st.items():
        if not s["m"]:
            continue
        ok = s["bad"] == 0 and s["emu"] < 1e-5 and s["dot"] < 1e-5 and s["avx2_bad"] == 0
        fails += not ok
        s_avx = f"{s['avx2_bad']} diferentes"
        print(f"{ty}_R{k} ({(BYTES[ty] + 32 * k) * 8 / 256:.2f} bits/peso): {s['m']} matrizes, {s['n']/1e6:.0f} M pesos | "
              f"C x float32 numpy: {s['bad']} diferentes | C x emulação float64: {s['emu']:.2e} do passo | "
              f"produto escalar: erro rel. {s['dot']:.2e} em {s['rows']} linhas"
              f"{f', AVX2 x escalar: {s_avx}' if HAS_AVX2 else ''} | "
              f"desquantização {s['n']/s['deq_s']/1e6:.0f} M pesos/s (escalar) {'OK' if ok else 'FALHOU'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
