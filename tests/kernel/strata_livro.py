#!/usr/bin/env python3
"""Livro de códigos aninhado (experimento, só qualidade; docs/09): o estrato 0 também é um livro de códigos não uniforme,
em vez de herdar a grade uniforme do Q4_K_M, e cada bit a mais divide cada centróide em dois filhos (Any-Precision LLM).

Dois desenhos, os dois ajustados com a imatrix (erro = Σ imatrix_j · (x_ij − q_ij)², o mesmo critério do llama-quantize):

--modo linha (Any-Precision / SqueezeLLM): cada linha da matriz tem o seu livro. O livro de --semente bits sai de k-means
    ponderado 1D na linha; cada nível seguinte divide cada célula em duas com 2-médias dentro da célula (a célula do pai
    fica fixa, então o código do nível k é 2·código do nível k − 1 + filho). Custo na RAM do nível de b bits: b bits por peso
    + uma tabela de 2^b valores F16 por linha (16·2^b/colunas bits por peso).
--modo bloco (híbrido estilo IQ4_XS): cada bloco de 32 pesos tem uma escala (6 bits com sinal × um F16 por 256 pesos,
    0,25 bit/peso, a conta do IQ4_XS) e o tensor tem um livro de 2^b valores (custo desprezível). O livro do nível 0 e as
    escalas saem de rodadas alternadas (escala por bloco com o livro fixo, Lloyd do livro com as escalas fixas), partindo
    da tabela do IQ4_NL; os níveis de cima dividem cada valor do livro em dois, por tensor (--filhos tensor) ou por linha
    (--filhos linha, tabela F16 por linha), com as escalas do nível 0 fixas.
--independente: sem aninhar; cada nível é ajustado do zero com 2^b valores (mede quanto o aninhamento custa).
--modo copia: só desquantiza as matrizes do --gguf para F16 (k-quants, IQ4_XS, Q8_0 no mesmo caminho de ativações).
--formato (modo bloco aninhado, --bits 4 5): grava os níveis no formato real dos tipos STRATA_LIVRO0/1 do patch de
    kernel/ggml (--out-tipo), com as tabelas em int8, e os GGUFs F16 de --out passam a ser a desquantização exata desse
    formato. Nível 0 = bloco do IQ4_XS byte a byte (d F16, escalas de 6 bits, códigos de 4 bits) com a tabela de 16
    valores do tensor; nível 1 = o mesmo bloco + um plano de 32 bytes (bit j do uint32 do bloco de 32 = peso j), tabela
    de 32 valores. As tabelas vão na chave strata.livro.<tensor> (int8). Uma escala α por tensor leva o livro para int8.
--fecho (modo bloco aninhado): estima quanto custa fechar o BF16 exato a partir de cada nível, como o fecho do arquivo
    0.3 (seção 21 de docs/05): quem leu o nível k sabe em que célula do livro o peso está, e o fecho escolhe o BF16 dentro
    dela ao custo de −log2 p(w) + log2 P(célula), com p = tabela de sinal+expoente por tensor e por oitava da escala do
    bloco, mantissa uniforme. Arquivo sem perdas lendo até o nível k = bits do nível k + fecho do nível k.

Sem kernel: os pesos de cada nível vão para um GGUF em F16, que o llama-perplexity lê direto, para medir a KL. Todas as
matrizes das camadas usam a mesma largura (sem a mistura do Q4_K_M, que daria mais bits às sensíveis); o resto do GGUF
(embedding, normas, saída) vem do --gguf.

    python tests/kernel/strata_livro.py --bf16 BF16.gguf --gguf Q4_K_M.gguf --imatrix imatrix.gguf \\
        --modo bloco --bits 4 5 6 --out n0.gguf n1.gguf n2.gguf
"""
import argparse, json, os, shutil, sys, tempfile, time

import numpy as np
import torch

GGUF_NAMES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float64) / 127


def ler_imatrix(path, gguf):
    """{nome do tensor: importância por coluna}, normalizada pelas contagens como o llama-quantize faz."""
    r = gguf.GGUFReader(path)
    t = {x.name: np.array(x.data, dtype=np.float32).reshape(-1) for x in r.tensors}
    out = {}
    for nome, s in t.items():
        if nome.endswith(".in_sum2"):
            base = nome[: -len(".in_sum2")]
            c = t[base + ".counts"]
            out[base] = s / c[0] if c[0] > 0 else np.ones_like(s)
    return out


# ---------------------------------------------------------------- modo linha: k-means 1D por linha, em lote

def media(w, x, c):
    return torch.where(w > 0, x / w.clamp_min(1e-300), c)


class Linhas:
    """Um lote de linhas ordenadas, com somas acumuladas por linha: a soma de qualquer intervalo da ordem sai em O(1), então
    cada rodada de Lloyd custa O(linhas × centróides), não O(pesos)."""

    def __init__(self, X, W):
        self.R, self.n = X.shape
        self.xs, self.perm = torch.sort(X, dim=1, stable=True)
        ws = torch.gather(W, 1, self.perm)
        z = torch.zeros(self.R, 1, dtype=X.dtype, device=X.device)
        self.CW = torch.cat([z, ws.cumsum(1)], 1)
        self.CX = torch.cat([z, (ws * self.xs).cumsum(1)], 1)
        self.CX2 = torch.cat([z, (ws * self.xs * self.xs).cumsum(1)], 1)
        self.ini = torch.zeros(self.R, 1, dtype=torch.long, device=X.device)
        self.fim = torch.full((self.R, 1), self.n, dtype=torch.long, device=X.device)

    def pos(self, b):
        """Índice do primeiro peso ≥ b em cada linha; b: (R, m)."""
        return torch.searchsorted(self.xs, b.contiguous())

    def somas(self, S, E):
        return self.CW.gather(1, E) - self.CW.gather(1, S), self.CX.gather(1, E) - self.CX.gather(1, S)

    def erro(self, C, S, E):
        """Σ w (x − c)² e Σ w x² do lote."""
        w, x = self.somas(S, E)
        x2 = self.CX2.gather(1, E) - self.CX2.gather(1, S)
        return float((x2 - 2 * C * x + C * C * w).sum()), float(self.CX2[:, -1].sum())

    def cortes(self, C):
        b = self.pos((C[:, :-1] + C[:, 1:]) / 2)
        return torch.cat([self.ini, b], 1), torch.cat([b, self.fim], 1)

    def kmeans(self, K, iters=30):
        """Livro de K valores por linha, partindo dos quantis ponderados. Devolve C, S, E (centróides e intervalos [S, E)
        de cada um na ordem)."""
        alvo = self.CW[:, -1:] * (torch.arange(K, dtype=self.xs.dtype, device=self.xs.device) + 0.5) / K
        i = (torch.searchsorted(self.CW.contiguous(), alvo.contiguous()) - 1).clamp(0, self.n - 1)
        C = self.xs.gather(1, i)
        for _ in range(iters):
            S, E = self.cortes(C)
            C = torch.sort(media(*self.somas(S, E), C), dim=1).values
        S, E = self.cortes(C)
        return media(*self.somas(S, E), C), S, E

    def divide(self, C, S, E, iters=12):
        """Cada célula [S, E) vira duas, com 2-médias dentro dela; devolve o livro de 2K valores (filhos intercalados)."""
        t = C.clone()
        for _ in range(iters + 1):
            m = torch.minimum(torch.maximum(self.pos(t), S), E)
            m0 = media(*self.somas(S, m), C)
            m1 = media(*self.somas(m, E), C)
            t = (m0 + m1) / 2
        R = self.R
        return (torch.stack([m0, m1], 2).reshape(R, -1), torch.stack([S, m], 2).reshape(R, -1),
                torch.stack([m, E], 2).reshape(R, -1))

    def monta(self, C, S, E):
        C = C.half().double()                                          # a tabela é guardada em F16
        v = torch.repeat_interleave(C.reshape(-1), (E - S).reshape(-1)).reshape(self.R, self.n)
        return torch.empty_like(v).scatter_(1, self.perm, v), C


def quantiza_linha(X, W, bits, semente, independente, lote):
    """Devolve [pesos do nível de cada largura em bits] e [Σ w e², Σ w x²] por nível."""
    outs = [torch.empty_like(X) for _ in bits]
    err = np.zeros((len(bits), 2))
    for a in range(0, X.shape[0], lote):
        L = Linhas(X[a:a + lote], W[a:a + lote])
        if independente:
            for i, b in enumerate(bits):
                C, S, E = L.kmeans(2 ** b)
                outs[i][a:a + L.R], Cq = L.monta(C, S, E)
                err[i] += L.erro(Cq, S, E)
            continue
        C, S, E = L.kmeans(2 ** semente)
        b = semente
        for i, alvo in enumerate(bits):
            while b < alvo:
                C, S, E = L.divide(C, S, E)
                b += 1
            outs[i][a:a + L.R], Cq = L.monta(C, S, E)
            err[i] += L.erro(Cq, S, E)
    return outs, err


# ---------------------------------------------------------------- modo bloco: escala por bloco + livro por tensor

def mais_perto(u, C):
    """Código do valor mais perto de u num livro crescente C (1D)."""
    return torch.bucketize(u, (C[:-1] + C[1:]) / 2)


def escalas(x, w, C, nf=5):
    """Escala por bloco (x, w: (blocos, 32)) que minimiza Σ w (x − s·C[q])², com q o mais perto de x/s: candidatos que
    levam o maior |x| do bloco a cada ponta do livro (os dois sinais), em ±12%, cada um refinado por mínimos quadrados."""
    j = x.abs().argmax(1, keepdim=True)
    xm = x.gather(1, j)
    xm = torch.where(xm == 0, torch.full_like(xm, 1e-30), xm)
    melhor_s = torch.ones_like(xm)
    melhor_e = torch.full_like(xm, float("inf"))
    for ponta in (C[0], C[-1]):
        for f in np.linspace(0.88, 1.12, nf):
            s = xm / (ponta * f)
            c = C[mais_perto(x / s, C)]
            den = (w * c * c).sum(1, keepdim=True)
            s = torch.where(den > 0, (w * x * c).sum(1, keepdim=True) / den.clamp_min(1e-300), s)
            s = torch.where(s == 0, torch.full_like(s, 1e-30), s)
            e = (w * (x - s * C[mais_perto(x / s, C)]) ** 2).sum(1, keepdim=True)
            m = e < melhor_e
            melhor_e = torch.where(m, e, melhor_e)
            melhor_s = torch.where(m, s, melhor_s)
    return melhor_s


def quantiza_escalas(s, partes=False):
    """Escala de 6 bits com sinal (−32 … 31, sem o 0) × um F16 por 8 blocos (256 pesos), como no IQ4_XS."""
    sb = s.reshape(-1, 8)
    d = (sb.abs().amax(1, keepdim=True) / 31).half().double().clamp_min(1e-30)
    l = torch.round(sb / d).clamp(-32, 31)
    l = torch.where(l == 0, torch.where(sb < 0, -1.0, 1.0).to(l.dtype), l)
    return (l, d) if partes else (l * d).reshape(-1, 1)


def lloyd_tensor(u, w, C, iters=8):
    K = len(C)
    for _ in range(iters):
        q = mais_perto(u, C).reshape(-1)
        sw = torch.bincount(q, w.reshape(-1), K)
        su = torch.bincount(q, (w * u).reshape(-1), K)
        C = torch.sort(media(sw, su, C)).values
    return C


def divide_grupos(u, w, C, g, G, iters=12):
    """Cada valor C[g] (um por grupo g: o código, ou linha·K + código) vira dois filhos por 2-médias dentro da célula;
    devolve o livro dos filhos (G, 2), o filho de cada peso e o corte que decidiu o filho (G,)."""
    gu, gw, uu = g.reshape(-1), w.reshape(-1), u.reshape(-1)
    t = C.reshape(-1).clone()
    for _ in range(iters + 1):
        corte = t
        f = uu >= t[gu]
        m1 = media(torch.bincount(gu, gw * f, G), torch.bincount(gu, gw * uu * f, G), t)
        m0 = media(torch.bincount(gu, gw * ~f, G), torch.bincount(gu, gw * uu * ~f, G), t)
        t = (m0 + m1) / 2
    return torch.stack([m0, m1], 1), f.reshape(u.shape).long(), corte


def erros_de(W, X, outs):
    return np.array([[float((W * (X - v) ** 2).sum()), float((W * X * X).sum())] for v in outs])


def formato_livro(x, s, u, C0, C1, R, n, out):
    """Níveis 0 e 1 no formato real: tabelas int8 (uma escala α por tensor, a mesma nos dois níveis, porque o bloco do
    nível 0 é compartilhado), códigos refeitos com as tabelas arredondadas e d = F16(d/α). Devolve os valores exatos."""
    l, d = quantiza_escalas(s, partes=True)                       # o s que veio já é l·d; aqui só se separa
    assert torch.equal((l * d).reshape(-1, 1), s), "escalas: l·d não volta ao s"
    alfa = 127 / float(torch.cat([C0, C1]).abs().max())
    T0, T1 = torch.round(C0 * alfa).clamp(-127, 127), torch.round(C1 * alfa).clamp(-127, 127)
    q0 = mais_perto(u, T0 / alfa)
    f = (u >= (T1[2 * q0] + T1[2 * q0 + 1]) / (2 * alfa)).long()
    q1 = 2 * q0 + f
    dst = (d / alfa).half()
    dl = (dst.double() * l).reshape(-1, 1)                         # escala de cada bloco de 32 como o kernel a vê
    out["valores"] = [(dl * T0[q0]).reshape(R, n), (dl * T1[q1]).reshape(R, n)]
    out["tab"] = [T0.cpu().numpy().astype(np.int8), T1.cpu().numpy().astype(np.int8)]
    # bloco do IQ4_XS por 256 pesos: d, scales_h (2 bits altos de cada ls), scales_l (4 bits baixos), qs
    nsb = l.shape[0]
    ls = (l + 32).cpu().numpy().astype(np.uint16)                   # (nsb, 8)
    qa = q0.cpu().numpy().astype(np.uint8).reshape(nsb, 8, 32)
    blk = np.zeros((nsb, 136), np.uint8)
    blk[:, 0:2] = dst.cpu().numpy().astype(np.float16).view(np.uint8).reshape(nsb, 2)
    sh = ((ls >> 4) << (2 * np.arange(8, dtype=np.uint16))).sum(1).astype(np.uint16)
    blk[:, 2:4] = sh.view(np.uint8).reshape(nsb, 2)
    blk[:, 4:8] = ((ls[:, 0::2] & 15) | ((ls[:, 1::2] & 15) << 4)).astype(np.uint8)
    blk[:, 8:136] = (qa[:, :, :16] | (qa[:, :, 16:] << 4)).reshape(nsb, 128)
    plano = (f.cpu().numpy().astype(np.uint32).reshape(nsb, 8, 32) << np.arange(32, dtype=np.uint32)).sum(2, dtype=np.uint32)
    out["blocos"] = [blk.reshape(R, -1), np.concatenate([blk, plano.view(np.uint8).reshape(nsb, 32)], 1).reshape(R, -1)]
    return out["valores"]


def quantiza_bloco(X, W, bits, filhos, independente, rodadas=3, ao_nivel=None, formato=None):
    """ao_nivel(i, s, lo, hi), se dado, recebe a célula de cada peso no nível i, no espaço u = x/s (lo/hi: (blocos, 32)).
    formato: um dict que recebe os blocos dos tipos STRATA_LIVRO0/1 e as tabelas int8; os valores devolvidos passam a ser
    a desquantização exata desse formato."""
    R, n = X.shape
    x, w = X.reshape(-1, 32), W.reshape(-1, 32)
    outs = []

    x32, w32 = x.float(), w.float()                                   # a busca das escalas em F32 (4× mais rápida)

    def ajusta(C):
        for r in range(rodadas):
            s = escalas(x32, w32, C.float()).double()
            if r == rodadas - 1:
                s = quantiza_escalas(s)
            C = lloyd_tensor(x / s, w * s * s, C)
        return s, C

    def erros():
        return erros_de(W, X, outs)

    iq4 = torch.tensor(IQ4NL, dtype=X.dtype, device=X.device)
    if independente:
        for b in bits:
            C0 = torch.tensor(np.interp(np.linspace(0, 15, 2 ** b), np.arange(16), IQ4NL), dtype=X.dtype, device=X.device)
            s, C = ajusta(C0)
            outs.append((s * C[mais_perto(x / s, C)]).reshape(R, n))
        return outs, erros()
    assert bits[0] == 4, "o modo bloco aninhado parte de 4 bits"
    s, C = ajusta(iq4)
    u, wu = x / s, w * s * s
    q = mais_perto(u, C)
    outs.append((s * C[q]).reshape(R, n))
    if formato is not None:
        assert list(bits) == [4, 5] and filhos == "tensor", "--formato: só os níveis 0 e 1 (--bits 4 5), filhos por tensor"
        F, _, _ = divide_grupos(u, wu, C, q, 16)
        return formato_livro(x, s, u, C, F.reshape(32), R, n, formato), erros_de(W, X, formato["valores"])
    if ao_nivel:
        inf = torch.tensor([float("inf")], dtype=C.dtype, device=C.device)
        M = torch.cat([-inf, (C[:-1] + C[1:]) / 2, inf])
        lo, hi = M[q], M[q + 1]
        ao_nivel(0, s, lo, hi)
    K = 16
    linha = (torch.arange(len(x), device=X.device) // (n // 32))[:, None]
    for b in bits[1:]:
        if filhos == "linha":
            g, G = linha * K + q, R * K
            Ccur = C.reshape(-1) if C.ndim > 1 else C.repeat(R)
        else:
            g, G, Ccur = q, K, C
        F, f, corte = divide_grupos(u, wu, Ccur, g, G)
        if ao_nivel:
            tq = corte[g]
            lo, hi = torch.where(f == 1, torch.maximum(lo, tq), lo), torch.where(f == 1, hi, torch.minimum(hi, tq))
            ao_nivel(len(outs), s, lo, hi)
        q, K = 2 * q + f, 2 * K
        if filhos == "linha":
            C = F.reshape(R, K).half().double()
            outs.append((s * C[linha, q]).reshape(R, n))
        else:
            C = F.reshape(K)
            outs.append((s * C[q]).reshape(R, n))
    return outs, erros()


# ---------------------------------------------------------------- fecho até o BF16 exato

_BF16 = None


def tabela_bf16():
    """Valores BF16 finitos em ordem crescente (um só zero) e o padrão de cada um."""
    global _BF16
    if _BF16 is None:
        allu = np.arange(65536, dtype=np.uint32)
        vals = (allu << 16).view(np.float32).astype(np.float64)
        fin = np.isfinite(vals) & (((allu & 0x7FFF) != 0) | (allu == 0))
        o = np.argsort(vals[fin], kind="stable")
        _BF16 = allu, vals[fin][o], allu[fin][o]
    return _BF16


def bits_fecho(pad, s, lo, hi):
    """Bits para escolher o BF16 exato (padrões pad, uint16) dentro da célula [s·lo, s·hi] de cada peso; contexto =
    oitava de |s| do bloco. Peso fora da célula (não deveria haver) paga −log2 p(w) inteiro e conta como escape."""
    allu, sv, ids = tabela_bf16()
    pad = pad.reshape(-1).astype(np.int64)
    pad = np.where(pad == 0x8000, 0, pad)
    w = (pad.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    s = s.expand(lo.shape).reshape(-1).cpu().numpy()
    a, b = (s * lo.reshape(-1).cpu().numpy()), (s * hi.reshape(-1).cpu().numpy())
    with np.errstate(invalid="ignore"):
        a, b = np.where(s < 0, b, a), np.where(s < 0, a, b)
    a = np.where(np.isnan(a), -np.inf, a)
    b = np.where(np.isnan(b), np.inf, b)
    eps = np.abs(s) * 1e-9
    a, b = a - eps, b + eps
    dentro = (w >= a) & (w <= b)
    ctx = np.clip(np.floor(np.log2(np.abs(s).clip(1e-30))).astype(np.int64), -60, 20) + 60
    usados, c = np.unique(ctx, return_inverse=True)
    se = np.zeros((len(usados), 512))
    np.add.at(se, (c, pad >> 7), 1.0)
    pv = se[:, allu >> 7] / 128.0 / se.sum(1, keepdims=True)
    cdf = np.concatenate([np.zeros((len(usados), 1)), np.cumsum(pv[:, ids], axis=1)], axis=1)
    i, j = np.searchsorted(sv, a, side="left"), np.searchsorted(sv, b, side="right")
    massa = cdf[c, j] - cdf[c, i]
    pw = pv[c, pad]
    bits = -np.log2(pw) + np.log2(np.where(dentro, np.maximum(massa, pw), 1.0))
    return float(bits.sum()), int((~dentro).sum())


# ---------------------------------------------------------------- GGUF

TIPO_LIVRO = [(51, 136), (52, 168)]                               # STRATA_LIVRO0/1 do patch de kernel/ggml: id, bytes/256


def escreve_tipos(gguf, reader, alvos, tabs, tmp, saidas, arch):
    """GGUFs dos tipos STRATA_LIVRO0/1: metadados e tensores de fora das camadas do --gguf, tabelas na chave
    strata.livro.<tensor>, blocos guardados em tmp por tensor."""
    for k, out in enumerate(saidas):
        tid, nb = TIPO_LIVRO[k]
        gguf.constants.GGML_QUANT_SIZES[tid] = (256, nb)
        wr = gguf.GGUFWriter(out, arch)
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            wr.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
        for name, tb in tabs.items():
            wr.add_key_value(f"strata.livro.{name}", [int(v) for v in tb[k]], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT8)
        for t, alvo in zip(reader.tensors, alvos):
            if alvo:
                rows, ncols = int(t.shape[1]), int(t.shape[0])
                wr.add_tensor_info(t.name, (rows, ncols // 256 * nb), np.dtype(np.uint8), rows * ncols // 256 * nb, raw_dtype=tid)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()
        for it, (t, alvo) in enumerate(zip(reader.tensors, alvos)):
            wr.write_tensor_data(np.load(f"{tmp}/{it}_{k}.npy") if alvo else np.array(t.data))
        wr.close()
    shutil.rmtree(tmp)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--gguf", required=True, help="GGUF de partida (metadados e tensores fora das camadas)")
    ap.add_argument("--imatrix")
    ap.add_argument("--modo", choices=["linha", "bloco", "copia"], required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 5, 6], help="largura de cada nível gravado")
    ap.add_argument("--semente", type=int, help="modo linha: bits do primeiro k-means (padrão: o primeiro de --bits)")
    ap.add_argument("--filhos", choices=["tensor", "linha"], default="tensor", help="modo bloco: livro dos níveis de cima")
    ap.add_argument("--independente", action="store_true", help="cada nível ajustado do zero (sem aninhar)")
    ap.add_argument("--camadas", type=int, help="só as primeiras N camadas (triagem rápida por erro; as outras saem do --gguf)")
    ap.add_argument("--lote", type=int, default=1024, help="modo linha: linhas por lote")
    ap.add_argument("--out", nargs="*", default=[], help="um GGUF F16 por nível (nenhum = só o erro ponderado)")
    ap.add_argument("--json", help="grava o resumo (bits por peso e erro ponderado por nível)")
    ap.add_argument("--fecho", action="store_true", help="modo bloco aninhado: estima o fecho até o BF16 exato a partir de cada nível")
    ap.add_argument("--formato", action="store_true", help="níveis 0 e 1 no formato real (tabelas int8); --out vira a desquantização exata")
    ap.add_argument("--out-tipo", nargs="*", default=[], help="com --formato: GGUF dos tipos STRATA_LIVRO0 e STRATA_LIVRO1 (um por nível)")
    args = ap.parse_args()
    if args.modo == "copia":
        args.bits = [0]
    if (args.formato or args.out_tipo) and (args.modo != "bloco" or args.bits != [4, 5] or args.independente or args.filhos != "tensor"):
        raise SystemExit("--formato/--out-tipo: modo bloco aninhado, --bits 4 5, filhos por tensor")
    if args.out_tipo and not args.formato:
        raise SystemExit("--out-tipo precisa de --formato")
    semente = args.semente or args.bits[0]

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

    def e_alvo(t):
        p = t.name.split(".")
        return (len(p) == 4 and p[0] == "blk" and p[2] in GGUF_NAMES and p[3] == "weight"
                and (args.camadas is None or int(p[1]) < args.camadas))

    alvos = [e_alvo(t) for t in reader.tensors]
    for t, alvo in zip(reader.tensors, alvos):
        for wr in writers:
            if alvo:
                rows = int(t.shape[1])
                ncols = int(t.shape[0])
                wr.add_tensor_info(t.name, (rows, ncols), np.dtype(np.float16), rows * ncols * 2, raw_dtype=gguf.GGMLQuantizationType.F16)
            else:
                wr.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    for wr in writers:
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_ti_data_to_file()

    if args.fecho and (args.modo != "bloco" or args.independente or args.filhos != "tensor"):
        raise SystemExit("--fecho só no modo bloco aninhado com filhos por tensor")
    n_pesos, bits_tab, err = 0, np.zeros(len(args.bits)), np.zeros((len(args.bits), 2))
    tabs = {}
    fecho, escapes = np.zeros(len(args.bits)), np.zeros(len(args.bits), dtype=np.int64)
    bytes_gguf, t0 = 0, time.time()
    tmp = tempfile.mkdtemp(prefix="livro_", dir=os.path.dirname(os.path.abspath(args.out_tipo[0]))) if args.out_tipo else None
    for it, (t, alvo) in enumerate(zip(reader.tensors, alvos)):
        if not alvo:
            for wr in writers:
                wr.write_tensor_data(np.array(t.data))
            continue
        rows, ncols = int(t.shape[1]), int(t.shape[0])
        n_pesos += rows * ncols
        bytes_gguf += int(t.n_bytes)
        X = (np.array(bf[t.name].data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, ncols)
        X = torch.from_numpy(X).to(DEV, torch.float64)
        qw = imat.get(t.name)
        W = torch.from_numpy(np.ones(ncols) if qw is None else qw.astype(np.float64)).to(DEV).expand(rows, ncols).contiguous()
        if args.modo == "copia":
            v = gguf.quants.dequantize(np.array(t.data), t.tensor_type).astype(np.float64).reshape(rows, ncols)
            vs = [torch.from_numpy(v).to(DEV)]
            e = np.array([[float((W * (X - vs[0]) ** 2).sum()), float((W * X * X).sum())]])
        elif args.modo == "linha":
            vs, e = quantiza_linha(X, W, args.bits, semente, args.independente, args.lote)
            for i, b in enumerate(args.bits):
                bits_tab[i] += rows * 16 * 2 ** b
        else:
            pad = np.array(bf[t.name].data).view(np.uint16).reshape(-1, 32)

            def ao_nivel(i, s, lo, hi):
                fb, esc = bits_fecho(pad, s, lo, hi)
                fecho[i] += fb
                escapes[i] += esc

            fmt = {} if args.formato else None
            vs, e = quantiza_bloco(X, W, args.bits, args.filhos, args.independente, ao_nivel=ao_nivel if args.fecho else None,
                                   formato=fmt)
            if tmp:
                for k in range(2):
                    np.save(f"{tmp}/{it}_{k}.npy", fmt["blocos"][k])
                tabs[t.name] = fmt["tab"]
            for i, b in enumerate(args.bits):
                bits_tab[i] += rows * ncols * 0.25 + (rows if args.filhos == "linha" and i > 0 else 1) * 16 * 2 ** b
                if args.fecho:
                    bits_tab[i] += sum(16 * 2 ** (c - 1) for c in args.bits[1:i + 1])   # cortes dos níveis 1 … i, F16
        err += e
        for wr, v in zip(writers, vs):
            wr.write_tensor_data(v.to(torch.float16).cpu().numpy())
        del X, W, vs
        print(f"  {t.name} {rows}×{ncols}: " + " ".join(f"{100 * a / b_:.3f}%" for a, b_ in e) + f"  ({time.time() - t0:.0f} s)",
              file=sys.stderr, flush=True)
    for wr in writers:
        wr.close()
    if tmp:
        escreve_tipos(gguf, reader, alvos, tabs, tmp, args.out_tipo, arch)
    res = {"modo": args.modo, "bits": args.bits, "semente": semente, "filhos": args.filhos, "independente": args.independente,
           "n_pesos": n_pesos, "niveis": []}
    for i, b in enumerate(args.bits):
        bpw = bytes_gguf * 8 / n_pesos if args.modo == "copia" else b + bits_tab[i] / n_pesos
        rel = err[i, 0] / err[i, 1]
        res["niveis"].append({"bits": b, "bits_por_peso": bpw, "erro_rel": rel})
        print(f"nível {i}: {bpw:.3f} bits/peso nas matrizes das camadas, erro ponderado relativo {100 * rel:.4f}%")
        if args.fecho:
            # lendo até o nível i: bits do nível i (planos crus de 1 bit, tabelas incluídas) + fecho até o BF16
            fpp = fecho[i] / n_pesos
            res["niveis"][-1].update({"fecho": fpp, "sem_perdas": bpw + fpp, "escapes": int(escapes[i])})
            print(f"   fecho até o BF16: {fpp:.3f} bits/peso, sem perdas {bpw + fpp:.3f} bits/peso, {escapes[i]} escapes")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
