#!/usr/bin/env python3
"""Tamanho dos níveis 1 e 2 do modo residual com os pesos saturados exatos já no nível 1 ou adiados para o último
nível (cortados na sub-célula da ponta nos níveis). Uso: python tests/ppl/strata_residual_adiar.py PASTA
onde PASTA tem model_path.txt (pasta dos safetensors) e Q4_K_M.gguf."""
import sys, os, numpy as np, torch
sys.path.insert(0, "tests/ppl")
import gguf
from safetensors.torch import load_file
from transformers import AutoConfig
from strata_ppl import GGUF_NAMES
from strata_residual import grid, level_bits
from strata_to_gguf import permute_like_gguf, w_to_u
from strata_residual import bf16_table
R = sys.argv[1]; path = open(R + "/model_path.txt").read().strip()
cfg = AutoConfig.from_pretrained(path); nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
sd = {}
for f in os.listdir(path):
    if f.endswith(".safetensors"): sd.update(load_file(os.path.join(path, f)))
_, vals, _, _ = bf16_table()
r4 = gguf.GGUFReader(R + "/Q4_K_M.gguf")
N = 0; base = 0.0; A = np.zeros(3); B = np.zeros(3)
for t in r4.tensors:
    p = t.name.split(".")
    if not (len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES): continue
    name = f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}"
    wt = sd[name + ".weight"]; u = wt.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
    w = permute_like_gguf(vals[u], name, tuple(wt.shape), nh, nkv); u = w_to_u(w)
    y, step, q, qmin, qmax, _ = grid(t, gguf)
    A += level_bits(u, y, step, q, qmin, qmax, 2)
    B += level_bits(u, y, step, q, qmin, qmax, 2, defer=True)
    N += u.size; base += 8 * int(t.n_bytes)
    b_ = base / N
    print(f"{t.name} parcial: exatos L1 {b_+A[0]/N:.3f} L2 {b_+(A[0]+A[1])/N:.3f} | adiados L1 {b_+B[0]/N:.3f} L2 {b_+(B[0]+B[1])/N:.3f}", flush=True)
b = base / N
print(f"base {b:.3f} | exatos: L1 {b+A[0]/N:.3f} L2 {b+(A[0]+A[1])/N:.3f} total {b+A.sum()/N:.3f} | adiados: L1 {b+B[0]/N:.3f} L2 {b+(B[0]+B[1])/N:.3f} total {b+B.sum()/N:.3f}")
