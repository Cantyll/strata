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
do grupo.
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
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.modelo)
    model = AutoModelForCausalLM.from_pretrained(args.modelo, torch_dtype=torch.bfloat16).to(args.device).eval()
    ids = tok(open(args.texto).read(), add_special_tokens=False)["input_ids"]
    n = min(args.chunks, len(ids) // args.ctx)

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
            np.savez(os.path.join(args.out, f"camada_{i}.npz"),
                     **{nome: (H[(i, nome)] / ntok).cpu().numpy() for nome in ("attn_in", "attn_out", "ffn_in", "ffn_down")})
        del H
    json.dump({"modelo": args.modelo, "texto": os.path.basename(args.texto), "trechos": n, "ctx": args.ctx,
               "tokens": ntok, "camadas": len(camadas)}, open(os.path.join(args.out, "info.json"), "w"))
    print(f"{n} trechos, {ntok} tokens, {time.time() - t0:.0f} s; Hessianas em {args.out}")


if __name__ == "__main__":
    main()
