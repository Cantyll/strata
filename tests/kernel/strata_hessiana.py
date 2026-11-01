#!/usr/bin/env python3
"""Hessianas das entradas de cada matriz das camadas (H = Σ x xᵀ), para o C1 de docs/10: escolher os bits dos níveis
com a covariância das ativações, como o GPTQ, no lugar do "filho mais próximo".

A imatrix do llama.cpp é a diagonal disto dividida pelo número de tokens. Aqui fica a matriz inteira, uma por entrada:
attn_in (q, k e v leem a mesma), attn_out (o), ffn_in (gate e up) e ffn_down. Mesmo texto e mesmos trechos da imatrix:
os primeiros --chunks trechos de --ctx tokens do arquivo, com o primeiro token de cada trecho trocado pelo BOS, como o
llama-imatrix faz. Ativações do modelo original (BF16), não do modelo já quantizado.

    python tests/kernel/strata_hessiana.py --modelo hf1b --texto wiki.train.raw --out hess/

Grava hess/camada_<i>.npz com as quatro matrizes (float32, já divididas pelo número de tokens) e hess/info.json.
Na CPU usa bf16 nas contas do modelo; na GPU (--device cuda), também. --grupo N junta as Hessianas de N camadas por
passada no texto (no 8B, as 32 camadas juntas não cabem na memória da GPU); cada passada para depois da última camada
do grupo. --fluxo faz uma passada só, camada a camada, com os estados de todos os trechos na GPU (1000 trechos do 8B em
fp16: 4,2 GB) e uma camada por vez; é o que cabe numa T4.
"""
import argparse, json, os, time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", required=True, help="pasta do modelo no formato do Hugging Face")
    ap.add_argument("--texto", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=125)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grupo", type=int, default=0, help="camadas por passada (0 = todas)")
    ap.add_argument("--fluxo", action="store_true",
                    help="camada a camada: uma passada só, todos os trechos juntos, uma camada por vez na GPU (8B numa T4)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="fp16 numa GPU sem bf16 (T4)")
    ap.add_argument("--triangulo", action="store_true",
                    help="grava só o triângulo de cima de cada H (metade do disco; strata_gptq_niveis.py refaz a matriz)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.modelo)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.modelo, torch_dtype=dt, attn_implementation="sdpa").eval()
    if not args.fluxo:
        model.to(args.device)
    ids = tok(open(args.texto).read(), add_special_tokens=False)["input_ids"]
    n = min(args.chunks, len(ids) // args.ctx)
    if args.fluxo:
        return fluxo(args, tok, model, ids, n)

    class Para(Exception):
        pass

    def gancho(H, i, nome):
        def f(_mod, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            h = H.get((i, nome))
            H[(i, nome)] = x.T @ x if h is None else h.add_(x.T @ x)
        return f

    def para(*_):
        raise Para

    os.makedirs(args.out, exist_ok=True)
    camadas = model.model.layers
    g = args.grupo or len(camadas)
    ntok = n * args.ctx
    t0 = time.time()
    for c0 in range(0, len(camadas), g):
        H, hooks = {}, []
        for i in range(c0, min(c0 + g, len(camadas))):
            layer = camadas[i]
            for nome, mod in (("attn_in", layer.self_attn.q_proj), ("attn_out", layer.self_attn.o_proj),
                              ("ffn_in", layer.mlp.gate_proj), ("ffn_down", layer.mlp.down_proj)):
                hooks.append(mod.register_forward_pre_hook(gancho(H, i, nome)))
        hooks.append(camadas[min(c0 + g, len(camadas)) - 1].register_forward_hook(para))
        with torch.no_grad():
            for c in range(n):
                x = list(ids[c * args.ctx:(c + 1) * args.ctx])
                x[0] = tok.bos_token_id
                try:
                    model(torch.tensor([x], device=args.device), use_cache=False)
                except Para:
                    pass
                if c % 25 == 0:
                    print(f"camadas {c0}-{min(c0 + g, len(camadas)) - 1}, trecho {c + 1}/{n}, {time.time() - t0:.0f} s", flush=True)
        for h in hooks:
            h.remove()
        for i in range(c0, min(c0 + g, len(camadas))):
            grava(os.path.join(args.out, f"camada_{i}.npz"),
                  {nome: H[(i, nome)] / ntok for nome in ("attn_in", "attn_out", "ffn_in", "ffn_down")}, args.triangulo)
        del H
    json.dump({"modelo": args.modelo, "texto": os.path.basename(args.texto), "trechos": n, "ctx": args.ctx,
               "tokens": ntok, "camadas": len(camadas)}, open(os.path.join(args.out, "info.json"), "w"))
    print(f"{n} trechos, {ntok} tokens, {time.time() - t0:.0f} s; Hessianas em {args.out}")


def grava(arq, H, triangulo):
    """Grava as Hessianas de uma camada (de forma atômica: quem lê nunca vê um arquivo pela metade)."""
    out = {}
    for nome in ("attn_in", "attn_out", "ffn_in", "ffn_down"):
        h = H[nome].cpu().numpy()
        out[nome] = np.concatenate([h[r, r:] for r in range(h.shape[0])]) if triangulo else h
    np.savez(arq + ".tmp.npz", **out)
    os.replace(arq + ".tmp.npz", arq)


def entradas(layer):
    return (("attn_in", layer.self_attn.q_proj), ("attn_out", layer.self_attn.o_proj),
            ("ffn_in", layer.mlp.gate_proj), ("ffn_down", layer.mlp.down_proj))


@torch.no_grad()
def fluxo(args, tok, model, ids, n, lote=8):
    """Uma passada no texto: os estados de todos os trechos ficam na GPU, e cada camada vai à GPU uma vez."""
    import copy
    dev, mm, t0 = args.device, model.model, time.time()
    mm.embed_tokens.to(dev); mm.rotary_emb.to(dev)
    vaga = copy.deepcopy(mm.layers[0]).to(dev)
    x = []
    for c in range(n):
        y = list(ids[c * args.ctx:(c + 1) * args.ctx])
        y[0] = tok.bos_token_id
        x.append(y)
    x = torch.tensor(x, device=dev)
    h = torch.cat([mm.embed_tokens(x[i:i + lote]) for i in range(0, n, lote)])
    pos = torch.arange(args.ctx, device=dev)[None]
    pe = mm.rotary_emb(h[:1], pos)
    H = {}

    def gancho(nome):
        def f(_mod, inp):
            z = inp[0].reshape(-1, inp[0].shape[-1]).float()
            H[nome] = z.T @ z if nome not in H else H[nome].add_(z.T @ z)
        return f

    os.makedirs(args.out, exist_ok=True)
    hooks = [mod.register_forward_pre_hook(gancho(nome)) for nome, mod in entradas(vaga)]
    for i, camada in enumerate(mm.layers):
        for a, b in zip(vaga.parameters(), camada.parameters()):
            a.copy_(b)
        H.clear()
        for j in range(0, n, lote):
            o = vaga(h[j:j + lote], position_embeddings=pe, position_ids=pos, attention_mask=None)
            h[j:j + lote] = o[0] if isinstance(o, tuple) else o
        grava(os.path.join(args.out, f"camada_{i}.npz"), {nome: H[nome] / (n * args.ctx) for nome in H}, args.triangulo)
        print(f"camada {i}: {time.time() - t0:.0f} s", flush=True)
    for hk in hooks:
        hk.remove()
    json.dump({"modelo": args.modelo, "texto": os.path.basename(args.texto), "trechos": n, "ctx": args.ctx,
               "tokens": n * args.ctx, "camadas": len(mm.layers), "fluxo": True, "dtype": args.dtype},
              open(os.path.join(args.out, "info.json"), "w"))
    print(f"{n} trechos, {n * args.ctx} tokens, {time.time() - t0:.0f} s; Hessianas em {args.out}")


if __name__ == "__main__":
    main()
