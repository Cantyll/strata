#!/usr/bin/env python3
"""Gera GGUFs com os níveis 1 e/ou 2 do modo residual em blocos montados (tipos STRATA_Q4_K_P<k>/STRATA_Q6_K_P<k>), para
rodar no llama.cpp com o patch de kernel/ggml (llama-bench, llama-perplexity, llama-cli).

Parte do GGUF Q4_K_M (estrato 0) e dos pesos BF16 originais. Cada matriz Q4_K ou Q6_K das camadas vira o bloco R<k> que o
arquivo .strata 0.3 guarda (bloco do GGUF + k planos, os mesmos de tests/kernel/test_q4kr.py) e em seguida o bloco
montado P<k>, que é o que o leitor faria ao carregar. O resto do GGUF (metadados, embeddings, normas) é copiado como está.
Com vários níveis, a célula de cada peso é calculada uma vez só e os GGUFs saem na mesma passada. As contas rodam em
PyTorch, na GPU quando houver (um 8B leva minutos na GPU e horas numa CPU de 4 núcleos).

    python tests/kernel/strata_gguf_niveis.py --model PASTA_OU_ID --gguf Q4_K_M.gguf --nivel 1 2 --out n1.gguf n2.gguf

Com --sub-celulas (vertente 3 de docs/07, escolhida na seção 28 de docs/05-resultados.md) saem os tipos STRATA_Q4_K_D<k> /
STRATA_Q6_K_D<k>: os mesmos blocos, mas cada filho de uma célula fica numa posição ajustada por tensor (3 rodadas de Lloyd
ponderado pela --imatrix), guardada numa tabela de 16 bytes na chave strata.tab.<tensor> do GGUF. Os deslocamentos
dependem só da classe do código do nível 0 (menor, maior ou meio) e dos bits dos níveis de baixo; o bit de cada nível é
o filho mais perto, não mais a metade da célula.

A célula nominal sai direto dos campos do bloco, com as mesmas operações em float32 do gguf-py (y = d·sc·q − dmin·mn no
Q4_K, y = d·sc·q no Q6_K) e o passo por sub-bloco; dá os mesmos índices que strata_residual.grid + nominal_cell
(conferido byte a byte nos GGUFs do Llama-3.2-1B, na CPU) sem desquantizar o tensor três vezes.
"""
import argparse, json, os, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))
sys.path.insert(0, os.path.join(ROOT, "tests", "kernel"))
from strata_ppl import GGUF_NAMES  # noqa: E402
from strata_residual import bf16_table  # noqa: E402
from strata_to_gguf import permute_like_gguf  # noqa: E402
from test_q4kr import BYTES, build_lib  # noqa: E402

# Números dos tipos no ggml com o patch de kernel/ggml (ggml.h) e bytes por bloco de 256 pesos.
LOTE = 1 << 16                                                # blocos de 256 pesos por lote (16 M pesos)
TIPOS = {("Q4_K", 1): (43, 176), ("Q4_K", 2): (44, 208), ("Q6_K", 1): (45, 242), ("Q6_K", 2): (46, 274)}
TIPOS_D = {chave: (tid + 4, nb) for chave, (tid, nb) in TIPOS.items()}   # sub-células por tensor: mesmos blocos
TAB_MARCA = 0x53


def _f16(raw, i):
    return raw[:, i:i + 2].contiguous().view(torch.float16).float()


def celula(raw, ty, codigo=False):
    """(lo, a, passo) da célula nominal de cada peso, formas (n, sub-blocos, pesos por sub-bloco): lo e a por peso em
    float64, passo com sinal por sub-bloco. raw é um tensor uint8 (n, bytes por bloco) em qualquer device; as contas são
    as do gguf-py, em float32 e uma operação por vez, antes de passar a float64. Com codigo=True devolve também o código
    de cada peso (0 … 15 no Q4_K, 0 … 63 no Q6_K), int64."""
    n = raw.shape[0]
    if ty == "Q4_K":
        d, dmin = _f16(raw, 0), _f16(raw, 2)
        e = raw[:, 4:16].reshape(n, 3, 4)                     # gguf.quants.Q4_K.get_scale_min
        sc = torch.cat([e[:, 0] & 0x3F, (e[:, 2] & 0x0F) | ((e[:, 0] >> 2) & 0x30)], dim=1)
        mn = torch.cat([e[:, 1] & 0x3F, (e[:, 2] >> 4) | ((e[:, 1] >> 2) & 0x30)], dim=1)
        dsc = (d * sc.float()).reshape(n, 8, 1)
        dm = (dmin * mn.float()).reshape(n, 8, 1)
        qs = raw[:, 16:144].reshape(n, 4, 1, 32)
        q = torch.cat([qs & 0x0F, qs >> 4], dim=2).reshape(n, 8, 32).float()
        y = (dsc * q - dm).double()
        passo = (dsc - dm).double() + dm.double()                 # deq(q = 1) − deq(q = 0)
        cod = q.to(torch.int64)
    else:
        d = _f16(raw, 208)
        dsc = (d * raw[:, 192:208].view(torch.int8).float()).reshape(n, 16, 1)
        ql = raw[:, 0:128].reshape(n, 2, 1, 64)
        ql = torch.cat([ql & 0x0F, ql >> 4], dim=2).reshape(n, 8, 32)
        qh = raw[:, 128:192].reshape(n, 2, 1, 32)
        qh = torch.cat([(qh >> s) & 0x03 for s in (0, 2, 4, 6)], dim=2).reshape(n, 8, 32)
        q = ((ql | (qh << 4)).to(torch.int16) - 32).reshape(n, 16, 16).float()
        y = (dsc * q).double()
        passo = (dsc * -31.0).double() - (dsc * -32.0).double()
        cod = q.to(torch.int64) + 32
    a = passo.abs()
    if codigo:
        return y - a / 2, a, passo, cod
    return y - a / 2, a, passo


def planos(w, lo, a, passo, k):
    """Os k planos de bits do nível k, (n, 32·k) uint8 no formato de test_q4kr.pack: a sub-célula de cada peso
    (0 … 2^k − 1, no sentido de q crescente, saturada na ponta), bit mais alto no primeiro plano."""
    m = 1 << k
    rel = torch.where(a > 0, (torch.minimum(torch.maximum(w, lo), lo + a) - lo) / a, torch.zeros_like(w))
    idx = torch.clamp(torch.floor(rel * m), 0, m - 1).to(torch.int64)
    idx = torch.where(passo < 0, m - 1 - idx, idx).reshape(-1, 32, 8)
    peso = (1 << torch.arange(8, device=w.device)).reshape(1, 1, 8)
    return torch.cat([(((idx >> (k - p)) & 1) * peso).sum(-1).to(torch.uint8) for p in range(1, k + 1)], dim=1)


def sub_celulas(x, v0, passo, cod, w, K, ncod, iters=3, pares_fixos=None):
    """Sub-células por tensor (tipos D). x, v0, w: (n, sub-blocos, pesos) float64; passo com sinal por sub-bloco; cod:
    código do nível 0 (0 … ncod − 1). No nível k, cada grupo (classe do código: 0 = menor, 1 = meio, 2 = maior; e os bits
    dos níveis 1 … k − 1) tem dois deslocamentos m₋, m₊ de 0 a 31, em unidades de |passo|/2^(k+4), e cada peso vai para o
    filho mais perto (3 rodadas de Lloyd ponderado por w, como strata_escala_nivel.niveis_desloc). Devolve as tabelas de
    16 bytes de cada nível 1 … K (int8, como em kernel/strata_q4kr.h) e o caminho r de cada peso no nível K (os K bits,
    nível 1 no mais alto, no sentido de q crescente) e os pares (m₊, m₋) de cada nível. Com pares_fixos, não ajusta: só
    leva cada peso ao filho mais perto com os pares dados (a segunda passada do gravador; o bincount na GPU soma em
    ordem qualquer, e refazer o ajuste poderia mudar um arredondamento)."""
    sg = torch.where(passo < 0, -1.0, 1.0).to(x.dtype)
    a = passo.abs()
    e = ((x - v0) * sg).reshape(-1)
    ws = w.reshape(-1)
    cl = torch.where(cod == 0, 0, torch.where(cod == ncod - 1, 2, 1)).reshape(-1)
    r = torch.zeros_like(cl)
    pares = []
    for k in range(1, K + 1):
        un = (a / 2 ** (k + 4)).expand_as(x).reshape(-1)
        g = (cl << (k - 1)) | r
        G = 3 << (k - 1)
        b = e >= 0
        wu, wu2 = ws * un, ws * un * un
        if pares_fixos is not None:
            mp, mm = (torch.from_numpy(v).to(x.device).to(x.dtype) for v in pares_fixos[k - 1])
            b = (e - mp[g] * un).abs() < (e + mm[g] * un).abs()
        for _ in range(0 if pares_fixos is not None else iters):
            bf = b.to(x.dtype)
            n1 = torch.bincount(g, bf, G)
            n0 = torch.bincount(g, 1 - bf, G)
            mp = torch.bincount(g, wu * e * bf, G) / torch.bincount(g, wu2 * bf, G).clamp_min(1e-300)
            mm = -torch.bincount(g, wu * e * (1 - bf), G) / torch.bincount(g, wu2 * (1 - bf), G).clamp_min(1e-300)
            mp = torch.where(n1 > 0, mp.round().clamp(0, 31), torch.full_like(mp, 8.0))
            mm = torch.where(n0 > 0, mm.round().clamp(0, 31), torch.full_like(mm, 8.0))
            b = (e - mp[g] * un).abs() < (e + mm[g] * un).abs()
        e = e - torch.where(b, mp[g], -mm[g]) * un
        r = 2 * r + b.long()
        pares.append((mp.long().cpu().numpy(), mm.long().cpu().numpy()))
    tabs = {}
    for k in range(1, K + 1):
        tab = np.zeros(16, np.int8)
        for c in range(3):
            for rr in range(1 << k):
                dl = 0
                for j in range(1, k + 1):                    # δ em unidades de 1/2^(k+4): o nível j vale 2^(k−j)
                    bit, sub = (rr >> (k - j)) & 1, rr >> (k - j + 1)
                    mp, mm = pares[j - 1]
                    gi = (c << (j - 1)) | sub
                    dl += (int(mp[gi]) if bit else -int(mm[gi])) << (k - j)
                tab[rr + (c << k)] = dl
        tab[15] = TAB_MARCA
        tabs[k] = tab
    return tabs, r.reshape(x.shape), pares


def conferir_d(lib, ty, k, tab, dst, r, v0, passo, cod):
    """Desquantiza os blocos D com a referência em C e compara com v0 + passo·δ/2^(k+4) em float64."""
    import ctypes
    f = lib.strata_q4kd_dequantize_row if ty == "Q4_K" else lib.strata_q6kd_dequantize_row
    f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    y = np.empty(r.numel(), np.float32)
    f(k, tab.ctypes.data, dst.ctypes.data, y.ctypes.data, r.numel())
    ncod = 16 if ty == "Q4_K" else 64
    cl = torch.where(cod == 0, 0, torch.where(cod == ncod - 1, 2, 1))
    dl = torch.from_numpy(tab.astype(np.float64)).to(r.device)[r + (cl << k)]
    v = (v0 + passo * dl / 2 ** (k + 4)).reshape(-1).cpu().numpy()
    err = np.abs(y - v).max() / max(np.abs(v).max(), 1e-30)
    if err > 1e-5:
        raise SystemExit(f"conferência falhou: {ty} nível {k}, erro relativo {err:.2e}")
    return err


def planos_de(r, k, K):
    """Os k planos do caminho r (calculado até o nível K), no formato de planos()."""
    idx = (r >> (K - k)).reshape(-1, 32, 8)
    peso = (1 << torch.arange(8, device=r.device)).reshape(1, 1, 8)
    return torch.cat([(((idx >> (k - p)) & 1) * peso).sum(-1).to(torch.uint8) for p in range(1, k + 1)], dim=1)


class Pesos:
    """Lê cada tensor do safetensors só quando pedido (um 8B não precisa caber inteiro na RAM)."""

    def __init__(self, path):
        from safetensors import safe_open
        self.arq = {}
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            mapa = json.load(open(idx))["weight_map"]
            for nome, f in mapa.items():
                self.arq[nome] = os.path.join(path, f)
        else:
            for f in sorted(os.listdir(path)):
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(path, f), "pt") as s:
                        for nome in s.keys():
                            self.arq[nome] = os.path.join(path, f)
        self.abertos = {}

    def __getitem__(self, nome):
        from safetensors import safe_open
        f = self.arq[nome]
        if f not in self.abertos:
            self.abertos[f] = safe_open(f, "pt")
        return self.abertos[f].get_tensor(nome)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--gguf", required=True, help="GGUF Q4_K_M (estrato 0)")
    ap.add_argument("--nivel", type=int, nargs="+", choices=[1, 2], required=True)
    ap.add_argument("--out", nargs="+", required=True, help="um GGUF por nível, na mesma ordem")
    ap.add_argument("--sub-celulas", action="store_true", help="tipos D: sub-células por tensor (precisa de --imatrix)")
    ap.add_argument("--imatrix", help="imatrix do Q4_K_M (pesa o ajuste das sub-células)")
    ap.add_argument("--conferir", action="store_true", help="tipos D: confere a desquantização do C contra a conta em float64")
    args = ap.parse_args()
    if len(args.out) != len(args.nivel):
        raise SystemExit("--out precisa de um arquivo por nível")
    if args.sub_celulas and not args.imatrix:
        raise SystemExit("--sub-celulas precisa de --imatrix")

    import gguf
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    tipos = TIPOS_D if args.sub_celulas else TIPOS
    for (_, _), (tid, nbytes) in tipos.items():               # o gguf-py não conhece os tipos novos
        gguf.constants.GGML_QUANT_SIZES[tid] = (256, nbytes)
    lib = build_lib()
    imat = {}
    if args.sub_celulas:
        from strata_base_fina import ler_imatrix
        imat = ler_imatrix(args.imatrix, gguf)
    path = args.model if os.path.isdir(args.model) else snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    cfg = AutoConfig.from_pretrained(path)
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads or cfg.num_attention_heads
    sd = Pesos(path)
    _, vals, _, _ = bf16_table()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vals = torch.from_numpy(vals).to(dev)

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

    def pesos_e_celulas(t):
        """BF16 do tensor na ordem do GGUF (n, 256) e os blocos do Q4_K_M."""
        p = t.name.split(".")
        name = f"model.layers.{p[1]}.{GGUF_NAMES[p[2]]}"
        wt = sd[name + ".weight"]
        u = wt.contiguous().view(torch.int16).numpy()
        return permute_like_gguf(u, name, tuple(wt.shape), n_head, n_kv).reshape(-1, 256), np.array(t.data)

    Kmax = max(args.nivel)

    def ajuste(t, pares_fixos=None):
        """Tabelas, caminhos e pares das sub-células do tensor t (tipos D), o tensor inteiro de uma vez."""
        ty = t.tensor_type.name
        u, data = pesos_e_celulas(t)
        raw = torch.from_numpy(data.reshape(-1, BYTES[ty])).to(dev)
        lo, a, passo, cod = celula(raw, ty, codigo=True)
        x = vals[torch.from_numpy(u).to(dev).long() & 0xFFFF].reshape(lo.shape)
        ncols = u.size // data.shape[0]
        qw = imat.get(t.name)
        qw = torch.ones(ncols, dtype=torch.float64) if qw is None else torch.from_numpy(qw.astype(np.float64))
        w = qw.to(dev).reshape(1, -1).expand(data.shape[0], -1).reshape(x.shape)
        tabs, r, pares = sub_celulas(x, lo + a / 2, passo, cod, w, Kmax, 16 if ty == "Q4_K" else 64, pares_fixos=pares_fixos)
        return tabs, r, pares, (lo + a / 2, passo, cod)

    # duas passadas: a primeira só declara os tensores (o cabeçalho do GGUF vem antes dos dados), a segunda calcula e grava
    alvos = []
    tabelas, pares_t = {}, {}
    for t in reader.tensors:
        p = t.name.split(".")
        ty = t.tensor_type.name
        alvo = len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and ty in BYTES
        alvos.append(alvo)
        if alvo and args.sub_celulas:
            tabelas[t.name], _, pares_t[t.name], _ = ajuste(t)
            for wr, k in zip(writers, args.nivel):
                wr.add_key_value(f"strata.tab.{t.name}", [int(v) for v in tabelas[t.name][k]], gguf.GGUFValueType.ARRAY,
                                 sub_type=gguf.GGUFValueType.INT8)
        for wr, k in zip(writers, args.nivel):
            if alvo:
                tid, pb = tipos[(ty, k)]
                rows, ncols = t.data.shape[0], t.data.shape[1] // BYTES[ty] * 256
                wr.add_tensor_info(t.name, (rows, ncols // 256 * pb), np.dtype(np.uint8), rows * (ncols // 256) * pb, raw_dtype=tid)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    for wr in writers:
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()

    n_w, n_bytes = 0, [0] * len(writers)
    for t, alvo in zip(reader.tensors, alvos):
        data = np.array(t.data)
        if not alvo:
            for wr in writers:
                wr.write_tensor_data(data)
            continue
        ty = t.tensor_type.name
        raw = data.reshape(-1, BYTES[ty])
        planes = {k: [] for k in args.nivel}
        if args.sub_celulas:
            tabs, r, _, (v0, passo, cod) = ajuste(t, pares_t[t.name])   # os pares da primeira passada
            assert all((tabs[k] == tabelas[t.name][k]).all() for k in args.nivel)
            u_size = r.numel()
            for k in args.nivel:
                planes[k].append(planos_de(r, k, Kmax).cpu().numpy())
        else:
            u, _ = pesos_e_celulas(t)
            u_size = u.size
            for i in range(0, raw.shape[0], LOTE):            # em lotes de blocos, para caber na memória da GPU
                rb = torch.from_numpy(raw[i:i + LOTE]).to(dev)
                lo, a, passo = celula(rb, ty)
                w = vals[torch.from_numpy(u[i:i + LOTE]).to(dev).long() & 0xFFFF].reshape(lo.shape)
                for k in args.nivel:
                    planes[k].append(planos(w, lo, a, passo, k).cpu().numpy())
        n_w += u_size
        for i, (wr, k) in enumerate(zip(writers, args.nivel)):
            blocks = np.ascontiguousarray(np.concatenate([raw] + [np.concatenate(planes[k])], axis=1))
            tid, pb = tipos[(ty, k)]
            dst = np.empty((data.shape[0], u_size // data.shape[0] // 256 * pb), dtype=np.uint8)
            (lib.strata_q4kr_to_q4kp if ty == "Q4_K" else lib.strata_q6kr_to_q6kp)(k, blocks.ctypes.data, dst.ctypes.data, u_size)
            if args.conferir and args.sub_celulas:
                conferir_d(lib, ty, k, tabs[k], dst, r >> (Kmax - k), v0, passo, cod)
            wr.write_tensor_data(dst)
            n_bytes[i] += dst.nbytes
    for wr in writers:
        wr.close()
    for out, k, nb in zip(args.out, args.nivel, n_bytes):
        print(f"{out}: nível {k}, {n_w / 1e6:.0f} M pesos nas matrizes das camadas, {8 * nb / n_w:.3f} bits/peso na RAM")


if __name__ == "__main__":
    main()
