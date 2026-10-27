#!/usr/bin/env python3
"""C1 de docs/10: os bits dos níveis escolhidos com a Hessiana das entradas (GPTQ), no lugar do filho mais próximo.

O nível 0 é o próprio --gguf (o Q4_K_M com imatrix do Hub), intacto. No nível k, cada peso tem dois filhos dentro da
célula do nível k − 1, e o arquivo guarda 1 bit por peso dizendo qual. Os pontos dos filhos vêm de --pontos:

- P: o meio de cada metade, v ± |passo|/2^(k+1); os pesos saem iguais aos dos tipos P (os de hoje);
- D: deslocamentos por código do nível anterior, por tensor, de 3 rodadas de Lloyd ponderado pela imatrix (tipos D,
  vertente 3 por tensor de docs/07; mesma conta de strata_escala_nivel.py --modo desloc --grupo tensor).

--escolha perto escolhe o filho mais próximo do peso original (o de hoje). --escolha gptq percorre as colunas (entradas
da matriz) na ordem do GPTQ, da maior para a menor diagonal da Hessiana; em cada coluna escolhe o filho mais próximo
do alvo já compensado e passa o erro para as colunas que faltam, pela inversa de H (a fatoração de Cholesky do
GPTQ). O alvo pode sair da célula do pai; aí fica o filho da ponta. O nível 2 parte dos valores do nível 1 já
escolhidos e recomeça do peso original.

Sem kernel: os pesos de cada nível vão para um GGUF com as matrizes das camadas em F16 (caminho F16 do llama-perplexity).
As Hessianas vêm de strata_hessiana.py. Imprime, por nível, o erro de saída Σ ΔW H ΔWᵀ somado nas matrizes, relativo ao
do nível 0, e grava esses números por tensor em --relatorio (JSON), se pedido.

    python tests/kernel/strata_gptq_niveis.py --bf16 BF16.gguf --gguf Q4_K_M.gguf --imatrix imatrix.gguf \\
        --hess hess/ --pontos P --escolha gptq --out n1.gguf n2.gguf
"""
import argparse, json, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
sys.path.insert(0, os.path.join(ROOT, "tests", "kernel"))
from strata_base_fina import ler_imatrix  # noqa: E402
from strata_gguf_niveis import celula  # noqa: E402
from strata_ppl import GGUF_NAMES  # noqa: E402
from test_q4kr import BYTES  # noqa: E402

ENTRADA = {"attn_q": "attn_in", "attn_k": "attn_in", "attn_v": "attn_in", "attn_output": "attn_out",
           "ffn_gate": "ffn_in", "ffn_up": "ffn_in", "ffn_down": "ffn_down"}


def perto(x, mais, menos):
    """True onde o filho 'mais' fica mais perto de x; no empate, o filho de valor maior (como o passo fixo de hoje)."""
    dm, dn = np.abs(x - mais), np.abs(x - menos)
    return np.where(dm == dn, mais >= menos, dm < dn)


def gptq(W, mais, menos, H, amortece=0.01, bloco=128, device="cpu"):
    """W, mais, menos: (linhas, colunas) float64 numpy; H: (colunas, colunas). Devolve o bit de cada peso (True =
    'mais'), escolhido coluna a coluna com o erro compensado pela Hessiana."""
    H = torch.from_numpy(np.array(H, dtype=np.float64)).to(device)
    n = H.shape[0]
    diag = torch.diagonal(H)
    morta = diag == 0
    H[morta, morta] = 1.0
    H += amortece * diag.mean() * torch.eye(n, dtype=H.dtype, device=device)
    perm = torch.argsort(torch.diagonal(H), descending=True)
    H = H[perm][:, perm]
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    W = torch.from_numpy(W).float().to(device)[:, perm].clone()
    A = torch.from_numpy(mais).float().to(device)[:, perm]
    B = torch.from_numpy(menos).float().to(device)[:, perm]
    b = torch.zeros(W.shape, dtype=torch.bool, device=device)
    for i1 in range(0, n, bloco):
        i2 = min(i1 + bloco, n)
        W1, E1, U1 = W[:, i1:i2].clone(), torch.zeros(W.shape[0], i2 - i1, device=device), U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w, a, c = W1[:, i], A[:, i1 + i], B[:, i1 + i]
            da, dc = (w - a).abs(), (w - c).abs()
            p = torch.where(da == dc, a >= c, da < dc)
            err = (w - torch.where(p, a, c)) / U1[i, i]
            W1[:, i:] -= err[:, None] * U1[i, i:][None, :]
            E1[:, i] = err
            b[:, i1 + i] = p
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    out = torch.empty_like(b)
    out[:, perm] = b
    return out.cpu().numpy()


def lloyd(x, v, sgs, un, c, w, iters=3):
    """Deslocamentos (0 … 31, em unidades de un) do filho de cima e do de baixo de cada código c, por tensor, como em
    strata_escala_nivel.niveis_desloc (por código, grupo tensor). Tudo achatado; devolve (mp, mm) por peso."""
    e = (x - v) * sgs
    G = int(c.max()) + 1
    b = e >= 0
    wu, wu2 = w * un, w * un * un
    for _ in range(iters):
        mp = np.bincount(c, wu * e * b, G) / np.maximum(np.bincount(c, wu2 * b, G), 1e-300)
        mm = -np.bincount(c, wu * e * ~b, G) / np.maximum(np.bincount(c, wu2 * ~b, G), 1e-300)
        mp = np.where(np.bincount(c, b.astype(np.float64), G) > 0, np.clip(np.rint(mp), 0, 31), 8.0)
        mm = np.where(np.bincount(c, (~b).astype(np.float64), G) > 0, np.clip(np.rint(mm), 0, 31), 8.0)
        b = np.abs(e - mp[c] * un) < np.abs(e + mm[c] * un)
    return mp[c], mm[c]


def erro_saida(d, H, device="cpu"):
    """Σ_linhas d H dᵀ para d (linhas, colunas)."""
    t = torch.from_numpy(d).float().to(device)
    return float(((t @ torch.from_numpy(np.asarray(H, dtype=np.float32)).to(device)) * t).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--gguf", required=True, help="Q4_K_M de partida (o nível 0)")
    ap.add_argument("--imatrix", help="pesos do Lloyd dos pontos D")
    ap.add_argument("--hess", required=True, help="pasta de strata_hessiana.py")
    ap.add_argument("--pontos", choices=["P", "D"], required=True)
    ap.add_argument("--escolha", choices=["perto", "gptq"], required=True)
    ap.add_argument("--amortece", type=float, default=0.01, help="amortecimento da diagonal de H (fração da média)")
    ap.add_argument("--out", nargs="+", required=True, help="um GGUF F16 por nível, 1 … K")
    ap.add_argument("--relatorio", help="JSON com o erro de saída por tensor e nível")
    ap.add_argument("--device", default="cpu", help="onde roda o GPTQ (cuda no 8B)")
    ap.add_argument("--cache", help="pasta para guardar os pesos de cada tensor já calculado e retomar depois de uma "
                    "interrupção (só vale para os mesmos argumentos)")
    args = ap.parse_args()
    K = len(args.out)
    torch.set_num_threads(os.cpu_count())
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    import gguf
    imat = ler_imatrix(args.imatrix, gguf) if args.imatrix else {}
    bf = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    reader = gguf.GGUFReader(args.gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writers = []
    for out in args.out:
        wr = gguf.GGUFWriter(out, arch)
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            wr.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
        writers.append(wr)
    alvos = []
    for t in reader.tensors:
        p = t.name.split(".")
        alvo = len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and t.tensor_type.name in BYTES
        alvos.append(alvo)
        for wr in writers:
            if alvo:
                rows = t.data.shape[0]
                ncols = t.data.shape[1] // gguf.GGML_QUANT_SIZES[t.tensor_type][1] * gguf.GGML_QUANT_SIZES[t.tensor_type][0]
                wr.add_tensor_info(t.name, (rows, ncols), np.dtype(np.float16), rows * ncols * 2, raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    for wr in writers:
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()

    camada, hcache = None, None
    soma = np.zeros(K + 1)
    rel = {}
    t0 = time.time()
    for t, alvo in zip(reader.tensors, alvos):
        data = np.array(t.data)
        if not alvo:
            for wr in writers:
                wr.write_tensor_data(data)
            continue
        cache = os.path.join(args.cache, t.name + ".npz") if args.cache else None
        if cache and os.path.exists(cache):                     # já calculado antes de um reinício
            z = np.load(cache)
            for k in range(K):
                writers[k].write_tensor_data(z[f"v{k + 1}"])
            rel[t.name] = [float(e) for e in z["erro"]]
            soma += np.array(rel[t.name])
            print(f"{t.name}: do cache", flush=True)
            continue
        p = t.name.split(".")
        if p[1] != camada:
            camada, hcache = p[1], np.load(os.path.join(args.hess, f"camada_{p[1]}.npz"))
        H = hcache[ENTRADA[p[2]]]
        ty = t.tensor_type.name
        rows = data.shape[0]
        raw = data.reshape(-1, BYTES[ty])
        u16 = np.array(bf[t.name].data).view(np.uint16).reshape(rows, -1)
        ncols = u16.shape[1]
        sb = 8 if ty == "Q4_K" else 16
        x = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(rows, ncols)
        v = gguf.quants.dequantize(data, t.tensor_type).astype(np.float64).reshape(rows, ncols)
        _, a, ps, cod = celula(torch.from_numpy(raw), ty, codigo=True)
        bloco = (a.shape[0], sb, 256 // sb)
        a = np.broadcast_to(a.numpy(), bloco).reshape(rows, ncols)
        sgs = np.sign(np.broadcast_to(ps.numpy(), bloco)).reshape(rows, ncols)
        sgs[sgs == 0] = 1.0
        c = cod.numpy().reshape(rows, ncols).astype(np.int64)
        qw = imat.get(t.name)
        w = np.ones_like(x) if qw is None else np.broadcast_to(qw.astype(np.float64), x.shape)
        e0 = erro_saida(x - v, H, args.device)
        soma[0] += e0
        rel[t.name] = [e0]
        saida = {}
        for k in range(1, K + 1):
            un = a / 2 ** (k + 4)
            if args.pontos == "P":
                mp = mm = 8.0
            else:
                mp, mm = lloyd(x.ravel(), v.ravel(), sgs.ravel(), un.ravel(), c.ravel(), w.ravel())
                mp, mm = mp.reshape(x.shape), mm.reshape(x.shape)
            mais, menos = v + sgs * mp * un, v - sgs * mm * un
            b = perto(x, mais, menos) if args.escolha == "perto" else gptq(x, mais, menos, H, args.amortece, device=args.device)
            v = np.where(b, mais, menos)
            c = 2 * c + b
            saida[f"v{k}"] = v.astype(np.float16)
            writers[k - 1].write_tensor_data(saida[f"v{k}"])
            ek = erro_saida(x - v, H, args.device)
            soma[k] += ek
            rel[t.name].append(ek)
        if cache:
            np.savez(cache + ".tmp.npz", erro=np.array(rel[t.name]), **saida)
            os.replace(cache + ".tmp.npz", cache)
        print(f"{t.name} ({ty}, {rows}×{ncols}): erro de saída / nível 0 = "
              + " ".join(f"{rel[t.name][k] / e0:.4f}" for k in range(1, K + 1)) + f"  [{time.time() - t0:.0f} s]", flush=True)
    for wr in writers:
        wr.close()
    print(f"pontos {args.pontos}, escolha {args.escolha}: erro de saída somado, relativo ao nível 0: "
          + " ".join(f"nível {k} {soma[k] / soma[0]:.4f}" for k in range(1, K + 1)))
    if args.relatorio:
        json.dump({"pontos": args.pontos, "escolha": args.escolha, "amortece": args.amortece, "soma": list(soma),
                   "tensores": rel}, open(args.relatorio, "w"))


if __name__ == "__main__":
    main()
