#!/usr/bin/env python3
"""ΔPPL real do STRATA: perplexidade no wikitext-2 com os pesos cortados em cada estrato,
contra o BF16 original e contra INT4/INT5/INT6/INT8 simétrico por bloco de 32.

O corte é o mesmo `truncMid` de src/strata-core.js (mantém sinal, expoente e os d primeiros
bits da mantissa; preenche o resto com o ponto médio, exceto em subnormais e inf/NaN).
Os INT-k seguem `blockQuantErr` do core: escala = max|x| / (2^(k-1) - 1) por bloco de 32
pesos contíguos na linha, arredondamento ao mais próximo, escala guardada em FP16.

Bits por peso em disco do STRATA são MEDIDOS: o script separa sinal, expoente e cada plano
da mantissa como o core (splitStrata) e comprime cada um com deflate-raw (zlib nível padrão,
o mesmo do CompressionStream). Para os INT-k o número é o nominal do formato: k + 16/32.

Uso típico (GPU ou CPU):
    pip install torch transformers datasets safetensors numpy
    huggingface-cli login            # Llama-3.2-1B é restrito: aceite a licença no HF antes
    python tests/ppl/strata_ppl.py --model meta-llama/Llama-3.2-1B
Rodada rápida para ver se tudo funciona:
    python tests/ppl/strata_ppl.py --model meta-llama/Llama-3.2-1B --max-chunks 4 --depths 2,7 --int-bits 4
"""
import argparse, json, math, os, subprocess, sys, time, zlib

import numpy as np
import torch

BF16_M, BF16_E = 7, 8


# ---------------- truncMid e INT-k (espelham src/strata-core.js) ----------------
def bf16_bits(w: torch.Tensor) -> torch.Tensor:
    """Padrões de bits BF16 (int32 em 0..65535) de um tensor BF16."""
    assert w.dtype == torch.bfloat16
    return w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def bits_to_bf16(u: torch.Tensor) -> torch.Tensor:
    u = u & 0xFFFF
    s = torch.where(u >= 0x8000, u - 0x10000, u)  # mesmo padrão de bits como int16 com sinal
    return s.to(torch.int16).view(torch.bfloat16)


def trunc_mid_bits(u: torch.Tensor, d: int) -> torch.Tensor:
    if d >= BF16_M:
        return u
    drop = BF16_M - d
    e = (u >> BF16_M) & ((1 << BF16_E) - 1)
    v = u & ~((1 << drop) - 1)
    mid = (e != 0) & (e != (1 << BF16_E) - 1)
    return torch.where(mid, v | (1 << (drop - 1)), v)


def int_block_quant(w: torch.Tensor, bits: int, blk: int = 32) -> torch.Tensor:
    """Simétrico por bloco de `blk` pesos contíguos na última dimensão; devolve float32 desquantizado."""
    levels = (1 << (bits - 1)) - 1
    x = w.float().reshape(-1)
    n = x.numel()
    pad = (-n) % blk
    if pad:
        x = torch.cat([x, x.new_zeros(pad)])
    xb = x.view(-1, blk)
    scale = (xb.abs().amax(dim=1, keepdim=True) / levels).half().float()  # escala guardada em FP16
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(xb / scale), -levels, levels) * scale
    return q.reshape(-1)[:n].view_as(w)


def prog_int_quant(w: torch.Tensor, k: int, blk: int = 32) -> torch.Tensor:
    """INT8 por bloco em sinal+magnitude, cortado nos k-1 planos de magnitude mais altos (ponto médio no resto).
    É o mesmo arquivo para todo k: baixar mais planos refina o modelo, como os estratos do STRATA."""
    x = w.float().reshape(-1, blk)
    scale = (x.abs().amax(dim=1, keepdim=True) / 127).half().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int32)
    m, drop = q.abs(), 8 - k
    mm = m.float() if drop == 0 else ((m >> drop) << drop).float() + (1 << (drop - 1))
    return (torch.where(q < 0, -mm, mm) * scale).reshape(-1)[:w.numel()].view_as(w)


def prog_int_stream_bytes(w: torch.Tensor, blk: int = 32) -> list:
    """Bytes de [escala FP16 + sinal, plano de magnitude 1, ..., 7], cada um com deflate-raw."""
    x = w.float().reshape(-1, blk)
    scale = (x.abs().amax(dim=1, keepdim=True) / 127).half()
    sf = torch.where(scale.float() == 0, torch.ones_like(scale.float()), scale.float())
    q = torch.clamp(torch.round(x / sf), -127, 127).to(torch.int32).reshape(-1)
    m = q.abs().numpy().astype(np.uint8)
    out = [deflate_len(scale.numpy().tobytes()) + deflate_len(np.packbits((q < 0).numpy().astype(np.uint8)).tobytes())]
    for j in range(7):
        out.append(deflate_len(np.packbits((m >> (6 - j)) & 1).tobytes()))
    return out


# ---------------- tamanho medido no disco (splitStrata + deflate-raw) ----------------
def deflate_len(b: bytes) -> int:
    c = zlib.compressobj(wbits=-15)
    return min(len(b), len(c.compress(b) + c.flush()))  # o core guarda cru quando deflate não ajuda


def strata_stream_bytes(u: np.ndarray) -> list:
    """Bytes por estrato de um tensor BF16: [sinal+expoente, plano 1, ..., plano 7]."""
    u = u.astype(np.uint16, copy=False).reshape(-1)
    sign = np.packbits(((u >> 15) & 1).astype(np.uint8))  # MSB primeiro, como o core
    exp = ((u >> BF16_M) & 0xFF).astype(np.uint8)
    out = [deflate_len(sign.tobytes()) + deflate_len(exp.tobytes())]
    for j in range(BF16_M):
        plane = np.packbits(((u >> (BF16_M - 1 - j)) & 1).astype(np.uint8))
        out.append(deflate_len(plane.tobytes()))
    return out


# ---------------- avaliação ----------------
def load_text(args):
    if args.text_file:
        return open(args.text_file, encoding="utf-8").read()
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


@torch.no_grad()
def perplexity(model, chunks, device):
    nll, ntok = 0.0, 0
    for ids in chunks:
        ids = ids.to(device)
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        nll += out.loss.float().item() * n
        ntok += n
    return math.exp(nll / ntok), ntok


def target_modules(model, include_embed):
    mods = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name.startswith("model.layers."):
            mods.append((name, m))
    if include_embed:
        mods.append(("model.embed_tokens", model.model.embed_tokens))
        if not model.config.tie_word_embeddings:
            mods.append(("lm_head", model.lm_head))
    return mods


def check_parity(repo_root):
    """Confere trunc_mid_bits contra o truncMid do core JS nos 65536 padrões BF16 (precisa de node)."""
    js = (
        "const fs=require('fs');new Function(fs.readFileSync(process.argv[1],'utf8'))();"
        "const S=globalThis.Strata,o=[];for(let d=0;d<=7;d++)for(let u=0;u<65536;u++)o.push(S.truncMid(u,S.LAYOUTS.BF16,d));"
        "process.stdout.write(Buffer.from(new Uint16Array(o).buffer))"
    )
    raw = subprocess.run(["node", "-e", js, os.path.join(repo_root, "src", "strata-core.js")], check=True, capture_output=True).stdout
    ref = np.frombuffer(raw, dtype=np.uint16).reshape(8, 65536)
    u = torch.arange(65536, dtype=torch.int32)
    for d in range(8):
        mine = trunc_mid_bits(u, d).numpy().astype(np.uint16)
        bad = int((mine != ref[d]).sum())
        if bad:
            raise SystemExit(f"truncMid diverge do core JS em d={d}: {bad} padrões")
    rt = bf16_bits(bits_to_bf16(u))
    if not torch.equal(rt, u):
        raise SystemExit("ida e volta int16<->bf16 não preserva os bits")
    print("paridade com src/strata-core.js: truncMid idêntico nos 65536 padrões BF16, d=0..7")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B", help="id no Hugging Face ou pasta local")
    ap.add_argument("--depths", default="0,1,2,3,4,5,6,7", help="estratos de mantissa a medir (7 = original)")
    ap.add_argument("--int-bits", default="4,5,6,8", help="baselines INT-k por bloco de 32 ('' para nenhum)")
    ap.add_argument("--prog-bits", default="", help="INT8 progressivo cortado em k bits, ex. '4,5,6,7' ('' para nenhum)")
    ap.add_argument("--ctx", type=int, default=2048, help="tokens por janela, sem sobreposição")
    ap.add_argument("--max-chunks", type=int, default=0, help="limita o número de janelas (0 = wikitext-2 inteiro)")
    ap.add_argument("--include-embed", action="store_true", help="também corta embed_tokens (e lm_head se não for amarrado)")
    ap.add_argument("--text-file", help="usar este texto em vez do wikitext-2 (só para teste do script)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compute-dtype", default="float32", choices=["float32", "bfloat16"],
                    help="dtype das contas; float32 isola o efeito dos pesos do arredondamento do BF16")
    ap.add_argument("--out", default="tests/ppl/resultados", help="pasta para JSON e tabela")
    ap.add_argument("--no-parity", action="store_true", help="pular a conferência contra o core JS")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not args.no_parity:
        check_parity(repo_root)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()
    mods = target_modules(model, args.include_embed)
    # bits BF16 originais, guardados uma vez na CPU
    orig = {name: bf16_bits(m.weight.data) for name, m in mods}
    n_target = sum(v.numel() for v in orig.values())
    n_total = sum(p.numel() for p in model.parameters())
    cdt = getattr(torch, args.compute_dtype)
    model.to(device=args.device, dtype=cdt)
    print(f"modelo {args.model}: {n_total/1e6:.1f} M parâmetros, {len(mods)} matrizes cortadas = {n_target/1e6:.1f} M "
          f"({100*n_target/n_total:.1f}%), device={args.device}, contas em {args.compute_dtype}, carga {time.time()-t0:.0f}s")

    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids
    nchunks = ids.shape[1] // args.ctx
    if args.max_chunks:
        nchunks = min(nchunks, args.max_chunks)
    chunks = [ids[:, i * args.ctx:(i + 1) * args.ctx] for i in range(nchunks)]
    print(f"texto: {ids.shape[1]} tokens, avaliando {nchunks} janelas de {args.ctx}")

    # tamanho medido do STRATA por estrato, só nas matrizes cortadas
    t1 = time.time()
    per_stratum = np.zeros(BF16_M + 1, dtype=np.int64)
    for name, u in orig.items():
        per_stratum += np.array(strata_stream_bytes(u.numpy()), dtype=np.int64)
    cum = np.cumsum(per_stratum)
    print(f"tamanhos medidos (deflate-raw) em {time.time()-t1:.0f}s")

    prog_bits = [int(x) for x in args.prog_bits.split(",") if x != ""]
    prog_cum = None
    if prog_bits:
        t1 = time.time()
        pb = np.zeros(8, dtype=np.int64)
        for name, u in orig.items():
            pb += np.array(prog_int_stream_bytes(bits_to_bf16(u)), dtype=np.int64)
        prog_cum = np.cumsum(pb)
        print(f"tamanhos medidos do INT8 progressivo em {time.time()-t1:.0f}s")

    def set_weights(fn):
        for name, m in mods:
            m.weight.data.copy_(fn(name).to(device=args.device, dtype=cdt))

    configs = []
    depths = [int(x) for x in args.depths.split(",") if x != ""]
    if BF16_M not in depths:
        depths.append(BF16_M)  # a referência BF16 sempre roda
    for d in sorted(depths, reverse=True):
        configs.append(("STRATA estrato %d" % d if d < BF16_M else "BF16 original (estrato 7)", "strata", d,
                        8 * cum[d] / n_target, 16.0,
                        lambda name, d=d: bits_to_bf16(trunc_mid_bits(orig[name], d))))
    for k in [int(x) for x in args.int_bits.split(",") if x != ""]:
        configs.append((f"INT{k} bloco 32", "int", k, k + 16 / 32, k + 16 / 32,
                        lambda name, k=k: int_block_quant(bits_to_bf16(orig[name]), k)))

    for k in prog_bits:
        configs.append((f"INT8 progressivo, {k} bits", "prog", k, 8 * prog_cum[k - 1] / n_target, k + 16 / 32,
                        lambda name, k=k: prog_int_quant(bits_to_bf16(orig[name]), k)))

    rows, base_ppl = [], None
    for label, kind, param, disk_bpw, ram_bpw, fn in configs:
        t2 = time.time()
        set_weights(fn)
        # erro relativo nos pesos (média ponderada por tensor, como depthMetrics)
        werr, wsum = 0.0, 0
        for name, m in mods:
            x = bits_to_bf16(orig[name]).double()
            y = fn(name).double()
            s = (x * x).sum().item()
            if s > 0:
                werr += x.numel() * math.sqrt(((y - x) ** 2).sum().item() / s)
                wsum += x.numel()
        ppl, ntok = perplexity(model, chunks, args.device)
        if kind == "strata" and param == BF16_M:
            base_ppl = ppl
        rows.append(dict(config=label, kind=kind, param=int(param), disk_bpw=float(disk_bpw), ram_bpw=float(ram_bpw),
                         weight_rel_err=werr / max(wsum, 1), ppl=ppl, tokens=ntok, seconds=time.time() - t2))
        print(f"{label:28s} disco {disk_bpw:6.3f} b/peso  erro pesos {100*rows[-1]['weight_rel_err']:7.3f}%  "
              f"PPL {ppl:9.4f}  ({time.time()-t2:.0f}s)", flush=True)

    for r in rows:
        r["delta_ppl"] = r["ppl"] - base_ppl
        r["delta_ppl_pct"] = 100 * (r["ppl"] / base_ppl - 1)

    os.makedirs(args.out, exist_ok=True)
    slug = args.model.strip("/").replace("/", "__")
    meta = dict(model=args.model, ctx=args.ctx, chunks=nchunks, tokens=rows[0]["tokens"], include_embed=args.include_embed,
                device=args.device, compute_dtype=args.compute_dtype, target_params=n_target, total_params=n_total,
                strata_bytes_per_stratum=per_stratum.tolist(), torch=torch.__version__)
    with open(os.path.join(args.out, f"{slug}.json"), "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2, ensure_ascii=False)

    lines = [f"Modelo `{args.model}`, {meta['tokens']} tokens do {'texto dado' if args.text_file else 'wikitext-2 (teste)'}, "
             f"janelas de {args.ctx}, contas em {args.compute_dtype}. Cortadas {len(mods)} matrizes = "
             f"{100*n_target/n_total:.1f}% dos parâmetros{' (com embedding)' if args.include_embed else ' (sem embedding nem normas)'}.",
             "", "| Configuração | Bits/peso no disco | Bits/peso na RAM | Erro rel. nos pesos | PPL | ΔPPL | ΔPPL % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: r["disk_bpw"]):
        lines.append(f"| {r['config']} | {r['disk_bpw']:.3f}{' (nominal)' if r['kind']=='int' else ''} | {r['ram_bpw']:.1f} | "
                     f"{100*r['weight_rel_err']:.3f}% | {r['ppl']:.4f} | {r['delta_ppl']:+.4f} | {r['delta_ppl_pct']:+.2f}% |")
    lines += ["", "STRATA: disco medido com deflate-raw por plano (como o core); na RAM o corte volta a BF16 (16 bits), sem kernel próprio.",
              "INT8 progressivo: um só arquivo INT8 por bloco de 32 em sinal+magnitude; cortar planos dá k bits. Disco medido (escala + sinal + planos, deflate-raw); RAM nominal k + 0,5.",
              "INT-k: disco e RAM nominais (k bits + escala FP16 a cada 32 pesos), sem medir compressão extra."]
    table = "\n".join(lines)
    with open(os.path.join(args.out, f"{slug}.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
