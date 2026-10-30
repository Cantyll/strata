#!/usr/bin/env python3
"""C2 de docs/10: quanto a KL da saída muda quando um grupo de matrizes sobe ou desce um nível, medido no modelo inteiro.

O alocador de hoje (tests/demanda/strata_mistura.py) escolhe o nível de cada matriz pelo erro com imatrix, que não vê o
efeito da camada na saída. Aqui o efeito é medido: o modelo do Hugging Face roda na GPU com os pesos dos níveis trocados
em memória (sem gravar um GGUF por configuração), e a KL é contra as log-probabilidades do modelo original, guardadas
uma vez na CPU.

    python tests/kernel/strata_dkl.py mede --modelo hf/ --niveis e0.gguf e1.gguf e2.gguf --texto a.raw b.raw \\
        --chunks 16 --out dkl.json
    python tests/kernel/strata_dkl.py plano --dkl dkl.json --q4km Q4_K_M.gguf --gib 5.48 --out plano.json
    python tests/kernel/strata_dkl.py avalia --modelo hf/ --niveis e0.gguf e1.gguf e2.gguf --texto a.raw b.raw \\
        --chunks 16 --plano plano.json [plano2.json ...]

Os níveis são GGUFs com as matrizes das camadas em F16: o nível 0 é o Q4_K_M desquantizado (strata_escala_nivel.py
--modo copia) e os níveis 1 e 2 vêm de strata_gptq_niveis.py. Os outros tensores (embeddings, saída) vêm do nível 0,
desquantizados, como no arquivo que roda.

Grupos: tipo de matriz (7) × faixa de --banda camadas. `mede` parte de todas as matrizes em --ref (1 por padrão, o
meio da escada) e mede, para cada grupo, a KL com o grupo um nível abaixo e um acima. `plano` supõe que os efeitos
somam e escolhe o nível de cada grupo que dá a menor KL prevista no orçamento (programação dinâmica em MiB), com os
mesmos bytes do strata_mistura (Q4_K_M + 32 bytes por bloco de 256 pesos por nível + --tabela por matriz e nível).
`avalia` mede a KL de planos inteiros, para conferir se a soma prevê a medida.

A KL aqui é a do caminho do PyTorch (fp16 na GPU), não a do llama.cpp; serve para comparar configurações entre si. A
régua usa a KL do llama-perplexity nos textos de teste, com o plano gravado por strata_mistura.py grava --plano.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
from strata_ppl import GGUF_NAMES  # noqa: E402

TIPOS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]


def hf_de(w, nome_hf, n_head, n_kv):
    """Desfaz a permutação de linhas de q_proj/k_proj do convert_hf_to_gguf (LlamaModel.permute)."""
    base = nome_hf.removesuffix(".weight")
    h = n_head if base.endswith("q_proj") else n_kv if base.endswith("k_proj") else None
    if h is None:
        return w
    r = w.shape[0]
    return w.reshape(h, r // h // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


class Modelo:
    def __init__(self, args):
        import gguf
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = args.device
        self.dtype = torch.float16 if args.device != "cpu" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(args.modelo)
        self.m = AutoModelForCausalLM.from_pretrained(args.modelo, torch_dtype=self.dtype).to(self.dev).eval()
        cfg = self.m.config
        self.n_head, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.n_cam = cfg.num_hidden_layers
        self.readers = [gguf.GGUFReader(p) for p in args.niveis]
        self.t = [{t.name: t for t in r.tensors} for r in self.readers]
        self.params = dict(self.m.named_parameters())
        self.nivel = {}
        # tensores fora das camadas: os do nível 0 (Q4_K_M), desquantizados
        for gn, hn in (("token_embd.weight", "model.embed_tokens.weight"), ("output.weight", "lm_head.weight")):
            if gn in self.t[0] and hn in self.params:
                t = self.t[0][gn]
                v = gguf.quants.dequantize(t.data, t.tensor_type).reshape(self.params[hn].shape)
                with torch.no_grad():
                    self.params[hn].copy_(torch.from_numpy(np.ascontiguousarray(v)).to(self.params[hn].dtype))
        self.matrizes = [f"blk.{i}.{ty}.weight" for i in range(self.n_cam) for ty in TIPOS]

    def hf(self, nome):
        p = nome.split(".")
        return f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}.weight"

    def poe(self, nome, k):
        """Põe a matriz nome (nome do GGUF) no nível k."""
        if self.nivel.get(nome) == k:
            return
        hn = self.hf(nome)
        v = np.asarray(self.t[k][nome].data, dtype=np.float16)
        v = hf_de(v.reshape(self.params[hn].shape), hn, self.n_head, self.n_kv)
        with torch.no_grad():
            self.params[hn].copy_(torch.from_numpy(np.ascontiguousarray(v)).to(self.params[hn].dtype))
        self.nivel[nome] = k

    def plano(self, niveis):
        for n in self.matrizes:
            self.poe(n, niveis[n])

    def trechos(self, textos, chunks, ctx):
        out = []
        for f in textos:
            ids = self.tok(open(f).read(), add_special_tokens=False)["input_ids"]
            for c in range(min(chunks, len(ids) // ctx)):
                x = list(ids[c * ctx:(c + 1) * ctx])
                x[0] = self.tok.bos_token_id
                out.append(x)
        return out

    @torch.no_grad()
    def logp(self, x):
        y = self.m(torch.tensor([x], device=self.dev), use_cache=False).logits[0].float()
        return torch.log_softmax(y, -1)

    @torch.no_grad()
    def kl(self, trechos, base):
        """KL média por token (a primeira metade de cada trecho fica de fora, como no llama-perplexity)."""
        s, n = 0.0, 0
        for x, b in zip(trechos, base):
            q = self.logp(x)[len(x) // 2:]
            p = b.to(self.dev).float()
            s += float((p.exp() * (p - q)).sum())
            n += p.shape[0]
        return s / n


def base_de(args, md, trechos):
    """Log-probabilidades do modelo original (pesos do Hugging Face, antes de qualquer troca), em fp16 na CPU."""
    t0 = time.time()
    base = [md.logp(x)[len(x) // 2:].half().cpu() for x in trechos]
    print(f"base: {len(trechos)} trechos, {time.time() - t0:.0f} s", flush=True)
    return base


def grupos_de(md, banda):
    g = {}
    for i in range(md.n_cam):
        for ty in TIPOS:
            g.setdefault(f"{ty}@{i // banda * banda}-{min(i // banda * banda + banda, md.n_cam) - 1}", []).append(
                f"blk.{i}.{ty}.weight")
    return g


def mede(args):
    md = Modelo(args)
    trechos = md.trechos(args.texto, args.chunks, args.ctx)
    base = base_de(args, md, trechos)
    grupos = grupos_de(md, args.banda)
    pesos = {n: int(md.t[0][n].n_elements) for n in md.matrizes}
    K = len(args.niveis) - 1
    md.plano({n: args.ref for n in md.matrizes})
    t0 = time.time()
    kl_ref = md.kl(trechos, base)
    print(f"todas no nível {args.ref}: KL {1000 * kl_ref:.3f} ×1000 ({time.time() - t0:.0f} s)", flush=True)
    res = {"ref": args.ref, "kl_ref": kl_ref, "banda": args.banda, "trechos": len(trechos), "textos": args.texto,
           "pesos": pesos, "grupos": {}}
    for g, membros in grupos.items():
        kls = {args.ref: kl_ref}
        for k in (args.ref - 1, args.ref + 1):
            if not 0 <= k <= K:
                continue
            for n in membros:
                md.poe(n, k)
            kls[k] = md.kl(trechos, base)
            for n in membros:
                md.poe(n, args.ref)
        res["grupos"][g] = {"membros": membros, "kl": {str(k): v for k, v in kls.items()}}
        print(f"{g}: " + " ".join(f"nível {k} {1000 * (v - kl_ref):+.3f}" for k, v in sorted(kls.items()) if k != args.ref)
              + f"  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)


def plano(args):
    d = json.load(open(args.dkl))
    ref, kl_ref = d["ref"], d["kl_ref"]
    base = os.path.getsize(args.q4km)
    lim = args.gib * 2**30
    MiB = 2**20
    # custo de cada grupo em cada nível, em bytes acima do Q4_K_M, e a KL prevista somando os efeitos
    opc = []
    for g, v in d["grupos"].items():
        n_pesos = sum(d["pesos"][n] for n in v["membros"])
        lista = []
        for k, kl in v["kl"].items():
            k = int(k)
            custo = k * (n_pesos // 256 * 32 + args.tabela * len(v["membros"]))
            lista.append((k, custo, kl - kl_ref))
        opc.append((g, sorted(lista)))
    cap = int((lim - base) // MiB)
    if cap < 0:
        raise SystemExit(f"orçamento {args.gib} GiB menor que o Q4_K_M")
    # programação dinâmica: melhor soma de ΔKL por MiB usado (custos arredondados para cima)
    INF = float("inf")
    melhor = np.full(cap + 1, INF)
    melhor[0] = 0.0
    escolha = []
    for g, lista in opc:
        novo = np.full(cap + 1, INF)
        arg = np.full(cap + 1, -1, dtype=np.int64)
        for j, (k, custo, dk) in enumerate(lista):
            c = math.ceil(custo / MiB)
            if c > cap:
                continue
            cand = np.full(cap + 1, INF)
            cand[c:] = melhor[:cap + 1 - c] + dk
            m = cand < novo
            novo[m], arg[m] = cand[m], j
        escolha.append(arg)
        melhor = novo
    fim = int(np.argmin(melhor))
    niveis_g, usado = {}, fim
    for (g, lista), arg in zip(reversed(opc), reversed(escolha)):
        j = int(arg[usado])
        k, custo, _ = lista[j]
        niveis_g[g] = k
        usado -= math.ceil(custo / MiB)
    niveis = {n: niveis_g[g] for g, v in d["grupos"].items() for n in v["membros"]}
    total = base + sum(c for (g, lista) in opc for (k, c, _) in lista if k == niveis_g[g])
    previsto = kl_ref + float(melhor[fim])
    cont = [sum(1 for v in niveis_g.values() if v == k) for k in range(3)]
    res = {"gib_orcamento": args.gib, "gib": total / 2**30, "kl_prevista": previsto, "grupos": niveis_g, "niveis": niveis}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.out}: {total / 2**30:.3f} GiB (orçamento {args.gib}); grupos nos níveis 0/1/2: {cont[0]}/{cont[1]}/{cont[2]}; "
          f"KL prevista {1000 * previsto:.3f} ×1000")


def avalia(args):
    md = Modelo(args)
    trechos = md.trechos(args.texto, args.chunks, args.ctx)
    base = base_de(args, md, trechos)
    out = {}
    for p in args.plano:
        pl = json.load(open(p))
        md.plano({n: int(pl["niveis"].get(n, 0)) for n in md.matrizes})
        out[p] = md.kl(trechos, base)
        prev = pl.get("kl_prevista")
        print(f"{p}: KL medida {1000 * out[p]:.3f} ×1000" + (f", prevista {1000 * prev:.3f} ({100 * (prev / out[p] - 1):+.1f}%)"
                                                           if prev else ""), flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nome in ("mede", "avalia"):
        a = sub.add_parser(nome)
        a.add_argument("--modelo", required=True, help="pasta do modelo original no formato do Hugging Face")
        a.add_argument("--niveis", nargs="+", required=True, help="GGUFs F16 dos níveis 0, 1, 2")
        a.add_argument("--texto", nargs="+", required=True, help="textos da medida (fora dos de teste)")
        a.add_argument("--chunks", type=int, default=16, help="trechos de cada texto")
        a.add_argument("--ctx", type=int, default=512)
        a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        a.add_argument("--out")
    m = sub.choices["mede"]
    m.add_argument("--ref", type=int, default=1, help="nível de partida de todas as matrizes")
    m.add_argument("--banda", type=int, default=4, help="camadas por grupo")
    sub.choices["avalia"].add_argument("--plano", nargs="+", required=True)
    p = sub.add_parser("plano")
    p.add_argument("--dkl", required=True)
    p.add_argument("--q4km", required=True, help="o GGUF Q4_K_M (nível 0): o tamanho de partida")
    p.add_argument("--gib", type=float, required=True)
    p.add_argument("--tabela", type=int, default=0, help="bytes por matriz e nível além dos planos (16 com tabela)")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"mede": mede, "plano": plano, "avalia": avalia}[args.cmd](args)


if __name__ == "__main__":
    main()
