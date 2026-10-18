#!/usr/bin/env python3
"""Profundidade de estrato POR TENSOR: em vez de cortar as 112 matrizes no mesmo estrato, escolhe um estrato d_t
para cada matriz sob um orçamento de bits por peso e mede o ΔPPL real no wikitext-2.

1. Tamanho: bytes medidos de cada estrato de cada matriz, iguais aos do strata_ppl.py (core v0.2: rANS no
   expoente, deflate-raw nos planos). Custo de cortar a matriz t no estrato d = sinal+expoente + planos 1..d.
2. Sensibilidade (proxy de Fisher diagonal): numa amostra do wikitext-2 de TREINO (não do teste), acumula
   sum_i g_i^2 * (w_i(d) - w_i)^2 por matriz e por estrato, com g o gradiente da loss de cada janela. É a
   aproximação de segunda ordem do aumento da loss usada em HAWQ/SqueezeLLM; só a ordem relativa importa.
3. Alocação: mochila de múltipla escolha resolvida por multiplicador de Lagrange (bisseção) e completada de
   forma gulosa até o orçamento. Cada matriz recebe um estrato entre --min-depth e 7 (7 = sem perdas).
4. Avaliação no teste: BF16, estratos uniformes de referência e as alocações, com a mesma perplexidade do
   strata_ppl.py. Os orçamentos padrão são os bits/peso medidos de Q4_K_M, Q5_K_M e Q6_K nas mesmas matrizes.

Uso (T4 no Colab, cerca de 30 min no wikitext-2 inteiro):
    python tests/ppl/strata_alloc.py --model unsloth/Llama-3.2-1B
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_ppl import (BF16_M, Bits16, bf16_bits, bits_to_bf16, check_parity, load_text, perplexity,  # noqa: E402
                        strata_stream_bytes, target_modules, trunc_mid_bits)

# bits/peso dos GGUF nas 112 matrizes, medidos na seção 13 (tests/ppl/resultados/unsloth__Llama-3.2-1B_gguf2.json)
KQUANT_BUDGETS = "Q4_K_M=4.80,Q5_K_M=5.66,Q6_K=6.56"
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_calib(args):
    if args.text_file:
        return open(args.text_file, encoding="utf-8").read()
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return "\n\n".join(ds["text"][:20000])  # sobra texto para muitas janelas de 2048


def fisher_sensitivity(model, mods, chunks, device):
    """sens[t, d] = soma sobre janelas de sum_i (g_i * (w_i(d) - w_i))^2, d = 0..7 (d = 7 dá zero)."""
    names = [n for n, _ in mods]
    sens = torch.zeros(len(mods), BF16_M + 1, dtype=torch.float64, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    hooks = []
    for t, (name, m) in enumerate(mods):
        w = m.weight
        w.requires_grad_(True)

        def hook(p, t=t):
            g = p.grad.float()
            x = p.data.float()
            u = bf16_bits(p.data.to(torch.bfloat16))  # os pesos são BF16 exatos guardados em float32
            for d in range(BF16_M):
                y = bits_to_bf16(trunc_mid_bits(u, d)).float()
                sens[t, d] += ((g * (y - x)) ** 2).sum().double()
            p.grad = None  # não guarda gradiente: a memória fica a dos pesos

        hooks.append(w.register_post_accumulate_grad_hook(hook))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # só liga o checkpointing; Llama não tem dropout ativo
    ntok = 0
    for ids in chunks:
        ids = ids.to(device)
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state
        n = ids.shape[1] - 1
        loss = 0.0
        for a in range(0, n, 256):  # loss em fatias, como perplexity(): logits de 128 k não cabem inteiros
            b = min(a + 256, n)
            logits = model.lm_head(h[:, a:b]).float()
            loss = loss + torch.nn.functional.cross_entropy(logits[0], ids[0, a + 1:b + 1], reduction="sum")
        (loss / n).backward()
        ntok += n
    for hk in hooks:
        hk.remove()
    model.gradient_checkpointing_disable()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return sens.cpu().numpy(), names, ntok


def allocate(sens, cost, budget_bits, min_depth):
    """Mochila de múltipla escolha: minimiza sum sens[t, d_t] com sum cost[t, d_t] <= budget_bits."""
    T = sens.shape[0]
    ds = np.arange(min_depth, BF16_M + 1)
    S, C = sens[:, ds], cost[:, ds]
    if C[:, 0].sum() > budget_bits:
        return None  # nem o estrato mínimo em tudo cabe

    def pick(lam):
        j = np.argmin(S + lam * C, axis=1)
        return j, C[np.arange(T), j].sum()

    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget_bits:  # lam grande => estratos rasos
        hi *= 4
    for _ in range(200):
        mid = (lo + hi) / 2
        if pick(mid)[1] > budget_bits:
            lo = mid
        else:
            hi = mid
    j, used = pick(hi)
    j = j.copy()
    while True:  # gasta a sobra no melhor ganho por bit
        best, gain = -1, 0.0
        for t in range(T):
            if j[t] + 1 < len(ds):
                dc = C[t, j[t] + 1] - C[t, j[t]]
                if used + dc <= budget_bits and dc > 0:
                    r = (S[t, j[t]] - S[t, j[t] + 1]) / dc
                    if r > gain:
                        best, gain = t, r
        if best < 0:
            break
        used += C[best, j[best] + 1] - C[best, j[best]]
        j[best] += 1
    return ds[j]


def summarize(names, depth):
    """Estrato médio por tipo de matriz e por camada (texto curto para a tabela)."""
    by_kind = {k: [] for k in KINDS}
    by_layer = {}
    for n, d in zip(names, depth):
        parts = n.split(".")
        by_kind.setdefault(parts[-1], []).append(d)
        by_layer.setdefault(int(parts[2]), []).append(d)
    hist = np.bincount(depth, minlength=BF16_M + 1)
    return dict(kind={k: float(np.mean(v)) for k, v in by_kind.items() if v},
                layer={k: float(np.mean(v)) for k, v in sorted(by_layer.items())},
                hist=hist.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B", help="id no Hugging Face ou pasta local")
    ap.add_argument("--budgets", default=KQUANT_BUDGETS, help="orçamentos 'rótulo=bits/peso,...' nas matrizes cortadas")
    ap.add_argument("--min-depths", default="0,1", help="estrato mínimo permitido; uma alocação por valor")
    ap.add_argument("--uniform", default="1,2,3", help="estratos uniformes de referência na mesma rodada ('' para nenhum)")
    ap.add_argument("--calib-chunks", type=int, default=16, help="janelas do wikitext-2 de treino para a sensibilidade")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=0, help="limita as janelas do teste (0 = wikitext-2 inteiro)")
    ap.add_argument("--text-file", help="texto local para calibração e teste (só para testar o script)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compute-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tag", default="alloc")
    ap.add_argument("--out", default="tests/ppl/resultados")
    ap.add_argument("--no-parity", action="store_true")
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
    mods = target_modules(model, False)
    orig = Bits16({name: m.weight.data.contiguous().view(torch.int16).clone() for name, m in mods})
    numel = np.array([orig[n].numel() for n, _ in mods], dtype=np.int64)
    n_target = int(numel.sum())
    cdt = getattr(torch, args.compute_dtype)
    model.to(device=args.device, dtype=cdt)
    print(f"modelo {args.model}: {len(mods)} matrizes, {n_target/1e6:.1f} M pesos, device={args.device}, carga {time.time()-t0:.0f}s")

    # 1. custo em bits de cada matriz em cada estrato
    t1 = time.time()
    per = np.array([strata_stream_bytes(orig[n].numpy()) for n, _ in mods], dtype=np.int64)  # [T, 8]
    cost = 8.0 * np.cumsum(per, axis=1)
    print(f"tamanhos medidos em {time.time()-t1:.0f}s; uniforme: " +
          ", ".join(f"E{d} {cost[:, d].sum()/n_target:.3f}" for d in range(BF16_M + 1)))

    # 2. sensibilidade no texto de treino
    t1 = time.time()
    cal = tok(load_calib(args), return_tensors="pt").input_ids
    ncal = min(args.calib_chunks, cal.shape[1] // args.ctx)
    cal_chunks = [cal[:, i * args.ctx:(i + 1) * args.ctx] for i in range(ncal)]
    sens, names, cal_tok = fisher_sensitivity(model, mods, cal_chunks, args.device)
    print(f"sensibilidade de Fisher em {ncal} janelas de treino ({cal_tok} tokens) em {time.time()-t1:.0f}s")

    # 3. alocações
    configs = [("BF16 original (estrato 7)", "bf16", None, np.full(len(mods), BF16_M))]
    for d in [int(x) for x in args.uniform.split(",") if x != ""]:
        configs.append((f"STRATA uniforme E{d}", "uniform", None, np.full(len(mods), d)))
    seen = {}
    for item in [x for x in args.budgets.split(",") if x]:
        label, b = item.split("=")
        for md in [int(x) for x in args.min_depths.split(",") if x != ""]:
            depth = allocate(sens, cost, float(b) * n_target, md)
            if depth is None:
                print(f"{label}: orçamento {b} abaixo do estrato {md} uniforme, pulado")
                continue
            key = depth.tobytes()
            if key in seen:
                print(f"{label} mín. E{md}: mesma alocação de {seen[key]}, pulado")
                continue
            seen[key] = f"{label} mín. E{md}"
            configs.append((f"STRATA por tensor @ {label} (mín. E{md})", "alloc", float(b), depth))

    # 4. avaliação no teste
    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids
    nchunks = ids.shape[1] // args.ctx
    if args.max_chunks:
        nchunks = min(nchunks, args.max_chunks)
    chunks = [ids[:, i * args.ctx:(i + 1) * args.ctx] for i in range(nchunks)]
    print(f"teste: {nchunks} janelas de {args.ctx}")

    slug = args.model.strip("/").replace("/", "__") + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.out, exist_ok=True)
    meta = dict(model=args.model, ctx=args.ctx, chunks=nchunks, calib_chunks=ncal, calib_tokens=cal_tok,
                device=args.device, compute_dtype=args.compute_dtype, target_params=n_target,
                names=names, numel=numel.tolist(), stratum_bytes=per.tolist(), fisher_sens=sens.tolist(),
                torch=torch.__version__)
    rows, base = [], None
    for label, kind, budget, depth in configs:
        t2 = time.time()
        for t, (name, m) in enumerate(mods):
            m.weight.data.copy_(bits_to_bf16(trunc_mid_bits(orig[name], int(depth[t]))).to(device=args.device, dtype=cdt))
        ppl, ntok = perplexity(model, chunks, args.device)
        if kind == "bf16":
            base = ppl
        bpw = float(cost[np.arange(len(mods)), depth].sum() / n_target)
        proxy = float(sens[np.arange(len(mods)), depth].sum())
        rows.append(dict(config=label, kind=kind, budget=budget, disk_bpw=bpw, fisher_proxy=proxy, ppl=ppl, tokens=ntok,
                         depth=depth.tolist(), summary=summarize(names, depth), seconds=time.time() - t2))
        print(f"{label:40s} disco {bpw:6.3f} b/peso  proxy {proxy:10.4g}  PPL {ppl:9.4f}  ({time.time()-t2:.0f}s)", flush=True)
        with open(os.path.join(args.out, f"{slug}.json"), "w") as f:
            json.dump(dict(meta=meta, rows=rows), f, indent=1, ensure_ascii=False)

    for r in rows:
        r["delta_ppl_pct"] = 100 * (r["ppl"] / base - 1)
    with open(os.path.join(args.out, f"{slug}.json"), "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1, ensure_ascii=False)

    lines = [f"Modelo `{args.model}`, {rows[0]['tokens']} tokens de teste, janelas de {args.ctx}, contas em {args.compute_dtype}. "
             f"Sensibilidade de Fisher em {cal_tok} tokens do wikitext-2 de treino.", "",
             "| Configuração | Bits/peso no disco | PPL | ΔPPL % | Proxy de Fisher | Estratos (E0..E7) | Estrato médio q k v o gate up down |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for r in sorted(rows, key=lambda r: r["disk_bpw"]):
        s = r["summary"]
        lines.append(f"| {r['config']} | {r['disk_bpw']:.3f} | {r['ppl']:.4f} | {r['delta_ppl_pct']:+.2f}% | {r['fisher_proxy']:.4g} | "
                     + " ".join(str(x) for x in s["hist"]) + " | " + " ".join("%.1f" % s["kind"].get(k, float("nan")) for k in KINDS) + " |")
    lines += ["", "Bits no disco medidos como no strata_ppl.py (core v0.2), somando sinal+expoente e os planos 1..d_t de cada matriz. "
              "O mapa completo de estratos por matriz está no JSON. Referência da seção 13: Q4_K_M 4,80 b = +4,07%, "
              "Q5_K_M 5,66 b = +1,25%, Q6_K 6,56 b = +0,49%."]
    table = "\n".join(lines)
    with open(os.path.join(args.out, f"{slug}.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
