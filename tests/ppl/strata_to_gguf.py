#!/usr/bin/env python3
"""Lê o arquivo residual do STRATA até o nível 1 e grava um GGUF comum, que o llama.cpp roda sem mudança.

No modo residual com estrato 0 = Q4_K_M, o nível 1 divide cada célula do Q4_K ao meio. O valor lido fica em
y4 +- d/4, que é um nível de uma grade de passo d/2 com o mínimo deslocado em d/4. Isso é um bloco Q5_K:

    Q5_K:  y = (d/2) * sc * q5 - dmin' * mn',   q5 = 2*q4 + bit do nível 1 (0..31)

A escala d/2 com o mesmo sc de 6 bits é exata. O mínimo novo (dmin*mn + d*sc/4) é requantizado em 6 bits por
sub-bloco, com um dmin' FP16 por super-bloco; é o único arredondamento da conversão. Pesos que saturavam nas
pontas da grade do Q4_K vêm exatos no nível 1 e recebem o q5 mais próximo. As matrizes Q6_K do Q4_K_M ficam
como estão (o nível 1 não é lido para elas), então o GGUF de saída tem a mesma mistura de tipos de um Q5_K_M.

O script usa os pesos BF16 só para tirar o bit do nível 1 de cada peso (o que um leitor tiraria do arquivo).
Saída: o GGUF e uma linha com os bits/peso de download (estrato 0 + nível 1 das matrizes Q4_K, medido como em
strata_residual.py) e de memória.

    python tests/ppl/strata_to_gguf.py --model unsloth/Llama-3.2-1B --gguf q4km.gguf --out nivel1_q5k.gguf
"""
import argparse, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_ppl import GGUF_NAMES  # noqa: E402
from strata_residual import bf16_table, grid, level_bits, nominal_cell, sub_index  # noqa: E402

QK = 256


def permute_like_gguf(w, name, shape, n_head, n_kv):
    """Mesma permutação de linhas de q_proj/k_proj que o convert_hf_to_gguf aplica (LlamaModel.permute)."""
    h = n_head if name.endswith("q_proj") else n_kv if name.endswith("k_proj") else None
    w = w.reshape(shape)
    if h is not None:
        w = w.reshape(h, 2, shape[0] // h // 2, *shape[1:]).swapaxes(1, 2).reshape(shape)
    return w.reshape(-1)


def pack_scales(sc, mn):
    """Inverso de gguf.quants.Q4_K.get_scale_min: 8 escalas e 8 mínimos de 6 bits em 12 bytes."""
    sc = sc.astype(np.uint8)
    mn = mn.astype(np.uint8)
    out = np.zeros((sc.shape[0], 12), dtype=np.uint8)
    out[:, 0:4] = sc[:, 0:4] | ((sc[:, 4:8] >> 4) << 6)
    out[:, 4:8] = mn[:, 0:4] | ((mn[:, 4:8] >> 4) << 6)
    out[:, 8:12] = (sc[:, 4:8] & 0x0F) | ((mn[:, 4:8] & 0x0F) << 4)
    return out


def q4k_to_q5k_level1(raw, w, gguf):
    """raw: blocos Q4_K (n, 144) na ordem do GGUF; w: pesos exatos na mesma ordem. Devolve blocos Q5_K (n, 176)."""
    n = raw.shape[0]
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n)
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n)
    sc, mn = gguf.quants.Q4_K.get_scale_min(raw[:, 4:16].copy())
    qs = raw[:, 16:144].reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    q4 = (qs & 0x0F).reshape(n, 8, 32).astype(np.int64)
    a = (d[:, None] * sc.astype(np.float32)).astype(np.float64)        # passo por sub-bloco (float32 como o ggml)
    m = (dmin[:, None] * mn.astype(np.float32)).astype(np.float64)
    y = a[:, :, None] * q4 - m[:, :, None]
    wb = w.reshape(n, 8, 32)
    qf = q4.astype(np.float64)
    lo, aa, _, _, inside = nominal_cell(y, np.broadcast_to(a[:, :, None], y.shape), qf, 0, 15, wb)
    bit = sub_index(wb, lo, aa, 1).astype(np.int64)

    d2 = (d / 2).astype(np.float16)
    m2 = m + a / 4
    dmin2 = (m2.max(1) / 63.0).astype(np.float16)
    dm = dmin2.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mn2 = np.clip(np.rint(np.where(dm[:, None] > 0, m2 / dm[:, None], 0)), 0, 63).astype(np.int64)
    a2 = d2.astype(np.float64)[:, None] * sc
    mq = dm[:, None] * mn2
    with np.errstate(divide="ignore", invalid="ignore"):
        q_esc = np.clip(np.rint(np.where(a2[:, :, None] > 0, (wb + mq[:, :, None]) / a2[:, :, None], 0)), 0, 31)
    q5 = np.where(inside, 2 * q4 + bit, q_esc).astype(np.uint8)

    out = np.zeros((n, 176), dtype=np.uint8)
    out[:, 0:2] = d2.reshape(n, 1).view(np.uint8)
    out[:, 2:4] = dmin2.reshape(n, 1).view(np.uint8)
    out[:, 4:16] = pack_scales(sc, mn2)
    for j in range(8):
        out[:, 16:48] |= ((q5[:, j, :] >> 4) & 1) << j
    lo4 = (q5 & 0x0F).reshape(n, 4, 2, 32)
    out[:, 48:176] = (lo4[:, :, 0, :] | (lo4[:, :, 1, :] << 4)).reshape(n, 128)
    return out, int((~inside).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--gguf", required=True, help="GGUF Q4_K_M (estrato 0)")
    ap.add_argument("--out", required=True, help="GGUF de saída")
    ap.add_argument("--no-sizes", action="store_true", help="não medir os bits do nível 1 (mais rápido)")
    args = ap.parse_args()

    import gguf
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    path = args.model if os.path.isdir(args.model) else snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(path)
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads or cfg.num_attention_heads
    sd = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(path, f)))
    _, vals, _, _ = bf16_table()

    reader = gguf.GGUFReader(args.gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(args.out, arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)

    t0 = time.time()
    tensors, n_target, bits_dl, bits_ram, esc, n_conv = [], 0, 0.0, 0.0, 0, 0
    err = [0.0, 0.0, 0.0]
    for t in reader.tensors:
        p = t.name.split(".")
        name = f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}" if len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES else None
        data, qtype = np.array(t.data), t.tensor_type
        if name is not None:
            n_target += int(t.n_elements)
            bits_dl += 8 * int(t.n_bytes)
            if qtype.name == "Q4_K":
                wt = sd[name + ".weight"]
                u = wt.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
                w = permute_like_gguf(vals[u], name, tuple(wt.shape), n_head, n_kv)
                rows = data.shape[0]
                blocks, e = q4k_to_q5k_level1(data.reshape(-1, 144), w, gguf)
                esc += e
                n_conv += int(t.n_elements)
                deq = gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType.Q5_K).astype(np.float64).reshape(-1)
                y4 = gguf.quants.dequantize(data.reshape(-1, 144), qtype).astype(np.float64).reshape(-1)
                err[0] += ((deq - w) ** 2).sum()
                err[1] += ((y4 - w) ** 2).sum()
                err[2] += (w * w).sum()
                if not args.no_sizes:
                    y, step, q, qmin, qmax, _ = grid(t, gguf)
                    bits_dl += level_bits(u_perm := w_to_u(w), y, step, q, qmin, qmax, 1)[0]
                data, qtype = blocks.reshape(rows, -1), gguf.GGMLQuantizationType.Q5_K
            bits_ram += 8 * data.nbytes
        tensors.append((t.name, data, qtype))
    for name, data, qtype in tensors:
        writer.add_tensor(name, data, raw_shape=data.shape, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{args.out}: {n_conv/1e6:.0f} M pesos Q4_K -> Q5_K ({esc} saturados vindos exatos), "
          f"download {'' if args.no_sizes else f'{bits_dl/n_target:.3f} '}b/peso, memória {bits_ram/n_target:.3f} b/peso "
          f"nas matrizes das camadas; erro rel. nas matrizes convertidas {100*np.sqrt(err[0]/err[2]):.3f}% "
          f"(Q4_K de origem {100*np.sqrt(err[1]/err[2]):.3f}%) ({time.time()-t0:.0f}s)")


def w_to_u(w):
    """Padrões BF16 de valores que já são BF16 exatos (para medir o nível 1 na ordem do GGUF)."""
    return torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


if __name__ == "__main__":
    main()
