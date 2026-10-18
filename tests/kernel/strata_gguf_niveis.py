#!/usr/bin/env python3
"""Gera um GGUF com os níveis 1 ou 2 do modo residual em blocos montados (tipos STRATA_Q4_K_P<k>/STRATA_Q6_K_P<k>), para
rodar no llama.cpp com o patch de kernel/ggml (llama-bench, llama-perplexity, llama-cli).

Parte do GGUF Q4_K_M (estrato 0) e dos pesos BF16 originais. Cada matriz Q4_K ou Q6_K das camadas vira o bloco R<k> que o
arquivo .strata 0.3 guarda (bloco do GGUF + k planos, os mesmos de tests/kernel/test_q4kr.py) e em seguida o bloco
montado P<k>, que é o que o leitor faria ao carregar. O resto do GGUF (metadados, embeddings, normas) é copiado como está.

    python tests/kernel/strata_gguf_niveis.py --model PASTA_OU_ID --gguf Q4_K_M.gguf --nivel 1 --out nivel1.gguf
"""
import argparse, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
sys.path.insert(0, os.path.join(ROOT, "tests", "kernel"))
from strata_ppl import GGUF_NAMES  # noqa: E402
from strata_residual import bf16_table, grid, nominal_cell, sub_index  # noqa: E402
from strata_to_gguf import permute_like_gguf  # noqa: E402
from test_q4kr import BYTES, build_lib, pack  # noqa: E402

# Números dos tipos no ggml com o patch de kernel/ggml (ggml.h) e bytes por bloco de 256 pesos.
TIPOS = {("Q4_K", 1): (43, 176), ("Q4_K", 2): (44, 208), ("Q6_K", 1): (45, 274), ("Q6_K", 2): (46, 274)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--gguf", required=True, help="GGUF Q4_K_M (estrato 0)")
    ap.add_argument("--nivel", type=int, choices=[1, 2], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gguf
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    for (_, _), (tid, nbytes) in TIPOS.items():               # o gguf-py não conhece os tipos novos
        gguf.constants.GGML_QUANT_SIZES[tid] = (256, nbytes)
    lib = build_lib()
    path = args.model if os.path.isdir(args.model) else snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(path)
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads or cfg.num_attention_heads
    sd = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(path, f)))
    _, vals, _, _ = bf16_table()
    k = args.nivel

    reader = gguf.GGUFReader(args.gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(args.out, arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        writer.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

    out, n_w, n_bytes = [], 0, 0
    for t in reader.tensors:
        p = t.name.split(".")
        ty = t.tensor_type.name
        data = np.array(t.data)
        if len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and ty in BYTES:
            name = f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}"
            wt = sd[name + ".weight"]
            u = wt.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
            w = permute_like_gguf(vals[u], name, tuple(wt.shape), n_head, n_kv)
            rows, ncols = data.shape[0], data.shape[1] // BYTES[ty] * 256
            y, step, q, qmin, qmax, _ = grid(t, gguf)
            lo, a, _, _, _ = nominal_cell(y, step, q, qmin, qmax, w)
            idx = sub_index(np.clip(w, lo, lo + a), lo, a, k).astype(np.int64)
            idx = np.where(step < 0, (1 << k) - 1 - idx, idx).reshape(-1, 256)
            blocks = pack(data.reshape(-1, BYTES[ty]), idx, k)
            tid, pb = TIPOS[(ty, k)]
            dst = np.empty((rows, ncols // 256 * pb), dtype=np.uint8)
            (lib.strata_q4kr_to_q4kp if ty == "Q4_K" else lib.strata_q6kr_to_q6kp)(k, blocks.ctypes.data, dst.ctypes.data, w.size)
            writer.add_tensor_info(t.name, dst.shape, dst.dtype, dst.nbytes, raw_dtype=tid)
            out.append(dst)
            n_w += w.size
            n_bytes += dst.nbytes
        else:
            writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, raw_dtype=t.tensor_type)
            out.append(data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for d in out:
        writer.write_tensor_data(d)
    writer.close()
    print(f"{args.out}: nível {k}, {n_w / 1e6:.0f} M pesos nas matrizes das camadas, {8 * n_bytes / n_w:.3f} bits/peso na RAM")


if __name__ == "__main__":
    main()
