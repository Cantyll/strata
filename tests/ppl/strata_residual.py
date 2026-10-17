#!/usr/bin/env python3
"""Modo residual (spec B.1): quanto custa um arquivo SEM PERDAS cujo primeiro estrato é um k-quant de verdade?

Estrato 0 = as matrizes do GGUF (Q4_K_M, Q5_K_M ou Q6_K do llama-quantize, bytes reais). O resto do arquivo
guarda o BF16 exato de cada peso dado o estrato 0. O k-quant arredonda cada peso ao nível mais próximo da sua
grade (y = d*q - m no Q4_K/Q5_K, y = d*q no Q6_K, com d e m por sub-bloco de 16 ou 32 pesos), então quem lê o
estrato 0 já sabe que o peso está na célula [y - d/2, y + d/2] (aberta para fora nos níveis extremos). O resíduo
só precisa escolher o valor BF16 dentro da célula:

    custo(w) = -log2 p(w) + log2 P(célula),   P(célula) = soma de p(v) para os BF16 v dentro da célula

com p um modelo a priori dos valores BF16 do tensor. Dois modelos:
  * sinal+expoente: p(v) = p(sinal, expoente) / 128, tabela de ordem 0 por tensor (a mesma que o rANS do core
    já guarda) e mantissa uniforme. Implementável como está, com custo de cabeçalho desprezível.
  * histograma: p(v) = frequência do padrão BF16 inteiro no tensor. Limite ideal, sem contar o cabeçalho.
Tudo é soma exata de -log2 das probabilidades (o que um codificador aritmético atinge a poucos bytes por tensor).

A grade de cada peso sai do próprio GGUF: desquantizar o bloco original dá y; desquantizar cópias com todos os
índices zerados e com todos em 1 dá o passo d de cada peso sem reimplementar o empacotamento das escalas.

Uso (CPU; ~1 min por GGUF no Llama-3.2-1B):
    python tests/ppl/strata_residual.py --model unsloth/Llama-3.2-1B --gguf Q4_K_M=q4km.gguf,Q5_K_M=q5km.gguf
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_ppl import GGUF_NAMES, strata_stream_bytes  # noqa: E402

# layout de ggml-common.h por bloco de 256 pesos: faixas de bytes zeradas (índice mínimo) e com índice mínimo + 1
LAYOUT = {
    "Q4_K": dict(size=144, zero=[(16, 144)], one=[(16, 144, 0x11)], qmin=0, qmax=15),    # d, dmin, scales[12], qs[128]
    "Q5_K": dict(size=176, zero=[(16, 176)], one=[(48, 176, 0x11)], qmin=0, qmax=31),    # d, dmin, scales[12], qh[32], qs[128]
    "Q6_K": dict(size=210, zero=[(0, 192)], one=[(0, 128, 0x11)], qmin=-32, qmax=31),    # ql[128], qh[64], scales[16], d
}


def grid(t, gguf):
    """(y, passo, q) por peso, na ordem do GGUF: y desquantizado, passo da grade (com sinal) e índice inteiro."""
    kind = t.tensor_type.name
    if kind not in LAYOUT:
        raise SystemExit(f"{t.name}: tipo {kind} não suportado")
    L = LAYOUT[kind]
    raw = np.array(t.data, dtype=np.uint8).reshape(-1, L["size"])
    deq = lambda r: gguf.quants.dequantize(r.reshape(t.data.shape), t.tensor_type).astype(np.float64).reshape(-1)
    z = raw.copy()
    for a, b in L["zero"]:
        z[:, a:b] = 0
    o = z.copy()
    for a, b, v in L["one"]:
        o[:, a:b] = v
    y, y0 = deq(raw), deq(z)
    step = deq(o) - y0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(step != 0, (y - y0) / step, 0.0)
    q = np.rint(f)
    bad = int((np.abs(f - q) > 1e-3).sum())
    return y, step, q + L["qmin"], L["qmin"], L["qmax"], bad


def unpermute(a, name, shape, n_head, n_kv):
    a = a.reshape(shape)
    h = n_head if name.endswith("q_proj") else n_kv if name.endswith("k_proj") else None
    if h is not None:
        a = a.reshape(h, shape[0] // h // 2, 2, *shape[1:]).swapaxes(1, 2).reshape(shape)
    return a.reshape(-1)

_TABLE = None


def bf16_table():
    """Valores BF16 finitos ordenados pelo valor real (um só zero), para somar p(v) numa faixa."""
    global _TABLE
    if _TABLE is None:
        allu = np.arange(65536, dtype=np.uint32)
        vals = torch.from_numpy(allu.astype(np.int32)).to(torch.int16).view(torch.bfloat16).float().numpy().astype(np.float64)
        fin = np.isfinite(vals) & ~((allu & 0x7FFF) == 0) | (allu == 0)
        order = np.argsort(vals[fin], kind="stable")
        _TABLE = (allu, vals, vals[fin][order], allu[fin][order])
    return _TABLE


def nominal_cell(y, step, q, qmin, qmax, w):
    """Célula nominal [lo, lo + a] de cada peso, se a célula do k-quant é aberta para baixo/cima, e se w cabe na nominal."""
    a = np.abs(step)
    lo = y - a / 2
    open_lo = ((q <= qmin) & (step > 0)) | ((q >= qmax) & (step < 0))
    open_hi = ((q >= qmax) & (step > 0)) | ((q <= qmin) & (step < 0))
    eps = a * 1e-9
    inside = (w >= lo - eps) & (w <= lo + a + eps)
    return lo, a, open_lo, open_hi, inside


def sub_index(w, lo, a, k):
    n = 1 << k
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(a > 0, (w - lo) / a, 0.0)
    return np.clip(np.floor(rel * n), 0, n - 1)


def level_bits(u, y, step, q, qmin, qmax, K):
    """Bits de cada nível de refinamento: nível k divide a célula nominal em 2^k partes iguais; pesos fora da
    célula nominal (saturados nos níveis extremos) ficam exatos já no nível 1. Último item: do nível K ao BF16
    exato. A soma é o resíduo inteiro (regra da cadeia), com o mesmo modelo sinal+expoente por oitava do passo."""
    allu, vals, sv, ids = bf16_table()
    w = vals[u]
    lo, a, open_lo, open_hi, inside = nominal_cell(y, step, q, qmin, qmax, w)
    ctx = np.clip(np.floor(np.log2(np.where(a > 0, a, 1e-30))).astype(np.int64), -40, 10) + 40
    used = np.unique(ctx)
    remap = np.zeros(51, dtype=np.int64)
    remap[used] = np.arange(used.size)
    c = remap[ctx]
    se_c = np.zeros((used.size, 512))
    np.add.at(se_c, (c, u >> 7), 1.0)
    pv_c = se_c[:, allu >> 7] / 128.0 / se_c.sum(1, keepdims=True)
    cdf_c = np.concatenate([np.zeros((used.size, 1)), np.cumsum(pv_c[:, ids], axis=1)], axis=1)
    pw = pv_c[c, u]

    def mass(l, h, last):
        i = np.searchsorted(sv, l, side="left")
        j = np.where(last, np.searchsorted(sv, h, side="right"), np.searchsorted(sv, h, side="left"))
        return cdf_c[c, j] - cdf_c[c, i]

    prev = mass(np.where(open_lo, -np.inf, lo), np.where(open_hi, np.inf, lo + a), np.ones_like(inside))
    out = []
    for k in range(1, K + 1):
        n = 1 << k
        idx = sub_index(w, lo, a, k)
        cur = mass(lo + idx / n * a, lo + (idx + 1) / n * a, idx == n - 1)
        cur = np.where(inside & (a > 0), np.maximum(cur, pw), pw)
        out.append(float((np.log2(prev) - np.log2(cur)).sum()))
        prev = cur
    out.append(float((np.log2(prev) - np.log2(pw)).sum()))
    return out


def reconstruct(w, y, step, q, qmin, qmax, k, exact_out=True):
    """Valor lido com o estrato 0 mais k níveis: centro da sub-célula; fora da célula nominal, o peso exato.
    exact_out=False corta esses pesos na sub-célula da ponta, como faz uma grade Q5_K."""
    if k == 0:
        return y
    lo, a, _, _, inside = nominal_cell(y, step, q, qmin, qmax, w)
    wc = w if exact_out else np.clip(w, lo, lo + a)
    idx = sub_index(wc, lo, a, k)
    mid = lo + (idx + 0.5) / (1 << k) * a
    return np.where(inside | (not exact_out), np.where(a > 0, mid, y), w)


def residual_bits(u, y, step, q, qmin, qmax):
    """Bits para fechar o BF16 exato dado o estrato 0, nos dois modelos a priori. u = padrões BF16 (uint16)."""
    # valores BF16 finitos ordenados pelo valor real, para somar p(v) numa faixa com cumsum + searchsorted
    allu = np.arange(65536, dtype=np.uint32)
    vals = torch.from_numpy(allu.astype(np.int32)).to(torch.int16).view(torch.bfloat16).float().numpy().astype(np.float64)
    fin = np.isfinite(vals) & ~((allu & 0x7FFF) == 0) | (allu == 0)  # um só zero (+0); -0 conta junto
    order = np.argsort(vals[fin], kind="stable")
    sv = vals[fin][order]
    ids = allu[fin][order]

    w = vals[u]
    a_ = np.abs(step)
    lo, hi = y - a_ / 2, y + a_ / 2
    # nível extremo: a célula se abre para fora (o quantizador satura ali)
    lo = np.where(((q <= qmin) & (step > 0)) | ((q >= qmax) & (step < 0)), -np.inf, lo)
    hi = np.where(((q >= qmax) & (step > 0)) | ((q <= qmin) & (step < 0)), np.inf, hi)
    eps = a_ * 1e-9
    lo, hi = lo - eps, hi + eps
    inside = (w >= lo) & (w <= hi)

    out = {}
    hist = np.bincount(u, minlength=65536).astype(np.float64)
    hist[0] += hist[0x8000]  # -0 vira +0 na tabela ordenada (mesmo valor real)
    se = np.bincount(u >> 7, minlength=512).astype(np.float64)
    models = {
        "sinal_expoente": se[allu >> 7] / 128.0,
        "histograma": hist,
    }
    for label, pv in models.items():
        pv = pv / pv.sum() if label == "histograma" else pv / se.sum()
        cdf = np.concatenate([[0.0], np.cumsum(pv[ids])])
        a = np.searchsorted(sv, lo, side="left")
        b = np.searchsorted(sv, hi, side="right")
        pc = cdf[b] - cdf[a]
        pw = pv[np.where(u == 0x8000, 0, u)]
        bits = -np.log2(pw) + np.log2(np.where(inside, pc, 1.0))  # fora da célula: paga o peso sem condição
        out[label] = float(bits.sum())
    # sinal+expoente condicionado ao passo da grade: uma tabela por oitava de |d| (o passo diz a escala local)
    ctx = np.floor(np.log2(np.where(a_ > 0, a_, 1e-30))).astype(np.int64)
    ctx = np.clip(ctx, -40, 10) + 40
    used = np.unique(ctx)
    remap = np.zeros(51, dtype=np.int64)
    remap[used] = np.arange(used.size)
    c = remap[ctx]
    se_c = np.zeros((used.size, 512))
    np.add.at(se_c, (c, u >> 7), 1.0)
    pv_c = se_c[:, allu >> 7] / 128.0 / se_c.sum(1, keepdims=True)  # [contextos, 65536]
    cdf_c = np.concatenate([np.zeros((used.size, 1)), np.cumsum(pv_c[:, ids], axis=1)], axis=1)
    a = np.searchsorted(sv, lo, side="left")
    b = np.searchsorted(sv, hi, side="right")
    pc = cdf_c[c, b] - cdf_c[c, a]
    pw = pv_c[c, u]
    bits = -np.log2(pw) + np.log2(np.where(inside, pc, 1.0))
    out["sinal_expoente_ctx"] = float(bits.sum())
    out["header_ctx_bytes"] = int(used.size * 2 + 3 * np.count_nonzero(se_c))
    out["fora_da_celula"] = int((~inside).sum())
    out["header_bytes"] = int(2 + 3 * np.count_nonzero(se))  # tabela sinal+expoente como no stream rans do core
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B", help="id no Hugging Face ou pasta com os safetensors")
    ap.add_argument("--gguf", required=True, help="'Q4_K_M=arq.gguf,...' (llama-quantize sem imatrix)")
    ap.add_argument("--out", default="tests/ppl/resultados")
    ap.add_argument("--limit", type=int, default=0, help="só as primeiras N matrizes (teste rápido)")
    ap.add_argument("--levels", type=int, default=0, help="também mede o tamanho dos níveis 1..K de refinamento")
    ap.add_argument("--ppl", action="store_true", help="mede o ΔPPL do estrato 0 + níveis 0..K (precisa de GPU para ser rápido)")
    ap.add_argument("--ref", default="", help="no modo --ppl, GGUFs de referência na mesma rodada, 'Q5_K_M=arq,...'")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--text-file", help="texto local em vez do wikitext-2 (teste do script)")
    ap.add_argument("--niveis", default="", help="no modo --ppl, só estes níveis, '0,1' (padrão 0..--levels)")
    ap.add_argument("--sem-saturados", action="store_true",
                    help="no modo --ppl, mede também cada nível com os pesos fora da célula cortados na ponta (sem guardá-los exatos)")
    ap.add_argument("--so-cortados", action="store_true", help="com --sem-saturados, mede só o BF16 e os níveis cortados")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.ppl:
        return ppl_main(args)

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
    names = [k[:-len(".weight")] for k in sd if k.startswith("model.layers.") and k.endswith("_proj.weight")]
    names.sort(key=lambda n: (int(n.split(".")[2]), n))
    if args.limit:
        names = names[:args.limit]
    u_all = {n: (sd[n + ".weight"].contiguous().view(torch.int16).numpy().view(np.uint16)) for n in names}
    n_target = sum(u.size for u in u_all.values())

    t0 = time.time()
    strata_lossless = 8 * sum(sum(strata_stream_bytes(u)) for u in u_all.values())
    print(f"{len(names)} matrizes, {n_target/1e6:.1f} M pesos; STRATA sem perdas {strata_lossless/n_target:.3f} b/peso ({time.time()-t0:.0f}s)")

    results = dict(model=args.model, target_params=n_target, strata_lossless_bpw=strata_lossless / n_target, configs={})
    for item in [x for x in args.gguf.split(",") if x]:
        label, gpath = item.split("=", 1)
        r = gguf.GGUFReader(gpath)
        tens = {}
        for t in r.tensors:
            p = t.name.split(".")
            if len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES:
                tens[f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}"] = t
        t1 = time.time()
        tot = dict(base=0.0, sinal_expoente=0.0, histograma=0.0, header=0.0, sinal_expoente_ctx=0.0, header_ctx=0.0, fora=0)
        types, by_type = {}, {}
        for n in names:
            t = tens[n]
            u = u_all[n].reshape(-1)
            shape = tuple(sd[n + ".weight"].shape)
            y, step, q, qmin, qmax, bad = grid(t, gguf)
            if bad:
                print(f"aviso: {n}: {bad} pesos fora da grade inteira", flush=True)
            y, step, q = (unpermute(a, n, shape, n_head, n_kv) for a in (y, step, q))
            res = residual_bits(u, y, step, q, qmin, qmax)
            if args.levels:
                lv = level_bits(u, y, step, q, qmin, qmax, args.levels)
                tot["levels"] = [x + v for x, v in zip(tot.get("levels", [0.0] * len(lv)), lv)]
            tot["base"] += 8 * int(t.n_bytes)
            tot["sinal_expoente"] += res["sinal_expoente"]
            tot["histograma"] += res["histograma"]
            tot["header"] += 8 * res["header_bytes"]
            tot["sinal_expoente_ctx"] += res["sinal_expoente_ctx"]
            tot["header_ctx"] += 8 * res["header_ctx_bytes"]
            tot["fora"] += res["fora_da_celula"]
            types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + u.size
            bt = by_type.setdefault(t.tensor_type.name, dict(n=0, base=0.0, resid=0.0, strata=0.0))
            bt["n"] += u.size
            bt["base"] += 8 * int(t.n_bytes)
            bt["resid"] += res["sinal_expoente_ctx"] + 8 * res["header_ctx_bytes"]
            bt["strata"] += 8 * sum(strata_stream_bytes(u))
        c = {k: v / n_target for k, v in tot.items() if k not in ("fora", "levels")}
        if "levels" in tot:
            c["levels_bpw"] = [v / n_target for v in tot["levels"]]
            c["levels_header"] = c["header_ctx"]
            cum = c["base"]
            for i, v in enumerate(c["levels_bpw"][:-1]):
                cum += v
                print(f"  nível {i+1}: +{v:.3f} -> {cum:.3f} b/peso (sem cabeçalho)", flush=True)
        c["fora_da_celula"] = tot["fora"]
        c["lossless_bpw"] = c["base"] + c["sinal_expoente"] + c["header"]
        c["lossless_bpw_histograma"] = c["base"] + c["histograma"]
        c["lossless_bpw_ctx"] = c["base"] + c["sinal_expoente_ctx"] + c["header_ctx"]
        c["types"] = {k: v / n_target for k, v in types.items()}
        c["by_type"] = {k: dict(share=v["n"] / n_target, base=v["base"] / v["n"], resid=v["resid"] / v["n"],
                                strata=v["strata"] / v["n"]) for k, v in by_type.items()}
        print("  por tipo: " + "; ".join(f"{k} {100*v['share']:.0f}%: base {v['base']:.3f} + resíduo {v['resid']:.3f} vs STRATA {v['strata']:.3f}"
                                         for k, v in c["by_type"].items()), flush=True)
        results["configs"][label] = c
        print(f"{label}: base {c['base']:.3f} + resíduo {c['sinal_expoente']+c['header']:.3f} = {c['lossless_bpw']:.3f} b/peso, com contexto {c['lossless_bpw_ctx']:.3f} "
              f"(histograma ideal {c['lossless_bpw_histograma']:.3f}; STRATA {results['strata_lossless_bpw']:.3f}); "
              f"fora da célula: {tot['fora']} ({time.time()-t1:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    slug = args.model.strip("/").replace("/", "__") + "_residual"
    with open(os.path.join(args.out, slug + ".json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    s = results["strata_lossless_bpw"]
    lines = [f"Modelo `{args.model}`, {len(names)} matrizes ({n_target/1e6:.1f} M pesos). Arquivo sem perdas = estrato 0 k-quant + resíduo até o BF16 exato.", "",
             "| Estrato 0 | Bits/peso do estrato 0 | Resíduo (sinal+expoente) | Total sem perdas | vs STRATA sem perdas | Total com histograma ideal |",
             "| --- | ---: | ---: | ---: | ---: | ---: |",
             f"| sinal+expoente (STRATA 0.2) | 3,6 | | {s:.3f} | | |"]
    for k, c in results["configs"].items():
        lines.append(f"| {k} | {c['base']:.3f} | {c['sinal_expoente']+c['header']:.3f} | {c['lossless_bpw']:.3f} | {c['lossless_bpw']-s:+.3f} | {c['lossless_bpw_histograma']:.3f} |")
    with open(os.path.join(args.out, slug + ".md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def ppl_main(args):
    """ΔPPL do estrato 0 (o GGUF) e dos níveis 1..K, contra o BF16, no wikitext-2 de teste."""
    import gguf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from strata_ppl import GGUFWeights, load_text, perplexity, target_modules
    label, gpath = args.gguf.split(",")[0].split("=", 1)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval()
    mods = target_modules(model, False)
    orig = {n: m.weight.data.contiguous().view(torch.int16).numpy().view(np.uint16).copy() for n, m in mods}
    shapes = {n: tuple(m.weight.shape) for n, m in mods}
    n_target = sum(v.size for v in orig.values())
    cfg = model.config
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads or cfg.num_attention_heads
    model.to(device=args.device, dtype=torch.float32)
    r = gguf.GGUFReader(gpath)
    tens = {}
    for t in r.tensors:
        p = t.name.split(".")
        if len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES:
            tens[f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}"] = t
    base_bpw = 8 * sum(int(tens[n].n_bytes) for n, _ in mods) / n_target

    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids
    nchunks = ids.shape[1] // args.ctx
    if args.max_chunks:
        nchunks = min(nchunks, args.max_chunks)
    chunks = [ids[:, i * args.ctx:(i + 1) * args.ctx] for i in range(nchunks)]
    _, vals, _, _ = bf16_table()

    def set_level(k, exact_out=True):
        fora = 0
        for n, m in mods:
            y, step, q, qmin, qmax, _ = grid(tens[n], gguf)
            y, step, q = (unpermute(a, n, shapes[n], n_head, n_kv) for a in (y, step, q))
            w = vals[orig[n].reshape(-1)]
            if k == 1:
                fora += int((~nominal_cell(y, step, q, qmin, qmax, w)[4]).sum())
            rec = reconstruct(w, y, step, q, qmin, qmax, k, exact_out)
            m.weight.data.copy_(torch.from_numpy(rec.reshape(shapes[n]).astype(np.float32)).to(args.device))
        return fora

    rows, base = [], None
    niveis = [int(x) for x in args.niveis.split(",")] if args.niveis else list(range(0, (args.levels or 3) + 1))
    configs = [("BF16 original", None, True)] + [(f"{label} + {k} nível(is)" if k else f"{label} (estrato 0)", k, True)
                                                  for k in niveis]
    if args.sem_saturados:
        cortados = [(f"{label} + {k} nível(is), saturados cortados", k, False) for k in niveis if k > 0]
        configs = (configs[:1] if args.so_cortados else configs) + cortados
    refs = [x for x in args.ref.split(",") if x]
    for label_k, k, exact_out in configs:
        t0 = time.time()
        if k is None:
            for n, m in mods:
                m.weight.data.copy_(torch.from_numpy(vals[orig[n].reshape(-1)].reshape(shapes[n]).astype(np.float32)).to(args.device))
            bpw, fora = 16.0, 0
        else:
            fora = set_level(k, exact_out)
            bpw = base_bpw + k  # nominal: ~1 bit por nível; o tamanho medido sai do modo sem --ppl com --levels
        ppl, ntok = perplexity(model, chunks, args.device)
        base = ppl if k is None else base
        rows.append(dict(config=label_k, level=k, bpw_nominal=bpw, ppl=ppl, tokens=ntok, fora_da_celula=fora))
        print(f"{label_k:28s} ~{bpw:6.3f} b/peso  PPL {ppl:9.4f}  ΔPPL {100*(ppl/base-1):+.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    for item in refs:
        rl, rp = item.split("=", 1)
        g = GGUFWeights(rp, cfg)
        t0 = time.time()
        for n, m in mods:
            m.weight.data.copy_(g(n, shapes[n]).to(args.device))
        bpw = 8 * sum(g.nbytes(n) for n, _ in mods) / n_target
        ppl, ntok = perplexity(model, chunks, args.device)
        rows.append(dict(config=f"llama.cpp {rl}", level=None, bpw_nominal=bpw, ppl=ppl, tokens=ntok))
        print(f"llama.cpp {rl:18s}  {bpw:6.3f} b/peso  PPL {ppl:9.4f}  ΔPPL {100*(ppl/base-1):+.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    for r_ in rows:
        r_["delta_ppl_pct"] = 100 * (r_["ppl"] / base - 1)
    os.makedirs(args.out, exist_ok=True)
    slug = args.model.strip("/").replace("/", "__") + "_residual_ppl"
    with open(os.path.join(args.out, slug + ".json"), "w") as f:
        json.dump(dict(model=args.model, gguf=label, chunks=nchunks, ctx=args.ctx, target_params=n_target, rows=rows), f, indent=1, ensure_ascii=False)
    lines = [f"Modelo `{args.model}`, {rows[0]['tokens']} tokens de teste, janelas de {args.ctx}, contas em float32. "
             f"Estrato 0 = {label} do llama-quantize; nível k = célula do k-quant dividida em 2^k partes, valor no centro.", "",
             "| Configuração | Bits/peso (nível: nominal) | PPL | ΔPPL % |", "| --- | ---: | ---: | ---: |"]
    for r_ in sorted(rows, key=lambda r_: r_["bpw_nominal"]):
        lines.append(f"| {r_['config']} | {r_['bpw_nominal']:.3f} | {r_['ppl']:.4f} | {r_['delta_ppl_pct']:+.2f}% |")
    with open(os.path.join(args.out, slug + ".md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
