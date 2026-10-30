#!/usr/bin/env python3
"""E5 da precisão sob demanda (docs/08): o modo "cabe no que tem". Cada matriz das camadas fica no nível 0, 1 ou 2 até
encher um orçamento de RAM, e a mistura é gravada em F16 para medir a KL no mesmo caminho de ativações dos k-quants
desquantizados (o "caminho F16" das seções 27–30 de docs/05).

Entrada: os GGUFs F16 de cada nível (o nível 0 é o Q4_K_M desquantizado, `strata_escala_nivel.py --modo copia`; os
níveis 1 e 2 vêm de `--modo fixo`, tipos P, ou `--modo desloc --grupo tensor --por borda`, tipos D), o BF16 e a imatrix.

    python tests/demanda/strata_mistura.py erros --bf16 BF16.gguf --imatrix imatrix.gguf --niveis e0.gguf e1.gguf e2.gguf \\
        --out erros.json
    python tests/demanda/strata_mistura.py grava --erros erros.json --q4km Q4_K_M.gguf --gib 5.8 \\
        --niveis e0.gguf e1.gguf e2.gguf --out mistura.gguf [--tabela 16]

A escolha: começa tudo no nível 0 e, enquanto couber, sobe o passo (matriz, nível) com o maior ganho de erro por
byte. O erro é Σ imatrix·(x − v)² com os próprios valores de cada nível (não uma estimativa). O passo do nível 2 só
entra depois do nível 1 da mesma matriz. O tamanho é o do GGUF Q4_K_M mais 32 bytes por bloco de 256 pesos para cada
nível (os planos), mais --tabela bytes por matriz e nível (16 nos tipos D, 0 nos P): é o que a RAM guarda com os tipos S.
A imatrix mede o peso de cada coluna pela ativação, não o efeito de cada camada na saída; um alocador melhor (Fisher,
seção 14 de docs/05) pode mudar a curva, e isso é dito no resultado.
"""
import argparse, heapq, json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "kernel"))
sys.path.insert(0, os.path.join(ROOT, "tests", "ppl"))


def matrizes(reader):
    """Os tensores que ganham níveis: as matrizes das camadas (blk.*) gravadas em F16 nos GGUFs de nível."""
    return [t for t in reader.tensors if t.name.startswith("blk.") and t.tensor_type.name == "F16" and len(t.shape) == 2]


def erros(args):
    import gguf
    from strata_base_fina import ler_imatrix
    imat = ler_imatrix(args.imatrix, gguf) if args.imatrix else {}
    bf = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    rs = [gguf.GGUFReader(p) for p in args.niveis]
    por_nome = [{t.name: t for t in r.tensors} for r in rs]
    out = {}
    for t in matrizes(rs[0]):
        x = (np.array(bf[t.name].data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        rows, ncols = x.shape
        qw = imat.get(t.name)
        w = np.ones(ncols, np.float64) if qw is None else qw.astype(np.float64)
        es = []
        for nm in por_nome:
            v = np.array(nm[t.name].data, dtype=np.float32).reshape(rows, ncols)
            es.append(float((((x - v).astype(np.float64) ** 2) * w).sum()))
        out[t.name] = {"pesos": rows * ncols, "erro": es}
        print(f"{t.name}: " + " ".join(f"{e:.4g}" for e in es), flush=True)
    json.dump(out, open(args.out, "w"))


def plano(err, gib, base_bytes, tabela):
    """Nível de cada matriz no orçamento gib (GiB). Devolve (níveis, bytes)."""
    nivel = {n: 0 for n in err}
    total = base_bytes
    lim = gib * 2**30
    heap = []

    def passo(n):
        k = nivel[n]
        if k >= len(err[n]["erro"]) - 1:
            return
        custo = err[n]["pesos"] // 256 * 32 + tabela
        ganho = err[n]["erro"][k] - err[n]["erro"][k + 1]
        heapq.heappush(heap, (-ganho / custo, n, k + 1, custo))

    for n in err:
        passo(n)
    while heap:
        _, n, k, custo = heapq.heappop(heap)
        if total + custo > lim:
            continue                                        # não cabe; um passo menor de outra matriz ainda pode caber
        nivel[n] = k
        total += custo
        passo(n)
    return nivel, total


def grava(args):
    import gguf
    err = json.load(open(args.erros))
    base_bytes = os.path.getsize(args.q4km)
    if args.plano:                      # plano pronto (tests/kernel/strata_dkl.py plano): só grava
        nivel = {n: int(k) for n, k in json.load(open(args.plano))["niveis"].items()}
        total = base_bytes + sum(nivel.get(n, 0) * (err[n]["pesos"] // 256 * 32 + args.tabela) for n in err)
    else:
        nivel, total = plano(err, args.gib, base_bytes, args.tabela)
    rs = [gguf.GGUFReader(p) for p in args.niveis]
    por_nome = [{t.name: t for t in r.tensors} for r in rs]
    r0 = rs[0]
    arch = r0.fields[gguf.Keys.General.ARCHITECTURE].contents()
    wr = gguf.GGUFWriter(args.out, arch)
    for field in r0.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        wr.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    escolhidos = []
    for t in r0.tensors:
        k = nivel.get(t.name, 0)
        src = por_nome[k][t.name]
        escolhidos.append(src)
        wr.add_tensor_info(src.name, src.data.shape, src.data.dtype, src.data.nbytes, raw_dtype=src.tensor_type)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_ti_data_to_file()
    for src in escolhidos:
        wr.write_tensor_data(np.array(src.data))
    wr.close()
    cont = [sum(1 for v in nivel.values() if v == k) for k in range(3)]
    pesos = [sum(err[n]["pesos"] for n, v in nivel.items() if v == k) for k in range(3)]
    tot_p = sum(pesos)
    res = {"gib_orcamento": args.gib, "gib": total / 2**30, "matrizes_por_nivel": cont,
           "fracao_pesos_por_nivel": [p / tot_p for p in pesos], "niveis": nivel}
    if args.plano_out:
        json.dump(res, open(args.plano_out, "w"))
    print(f"{args.out}: {total / 2**30:.3f} GiB (orçamento {args.gib}); matrizes nos níveis 0/1/2: {cont[0]}/{cont[1]}/{cont[2]} "
          f"({100 * pesos[0] / tot_p:.0f}/{100 * pesos[1] / tot_p:.0f}/{100 * pesos[2] / tot_p:.0f}% dos pesos)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("erros")
    a.add_argument("--bf16", required=True)
    a.add_argument("--imatrix")
    a.add_argument("--niveis", nargs="+", required=True, help="GGUFs F16 dos níveis 0, 1, 2")
    a.add_argument("--out", required=True)
    g = sub.add_parser("grava")
    g.add_argument("--erros", required=True)
    g.add_argument("--q4km", required=True, help="o GGUF Q4_K_M (nível 0): o tamanho de partida")
    g.add_argument("--gib", type=float, required=True, help="orçamento de RAM dos pesos, GiB")
    g.add_argument("--tabela", type=int, default=0, help="bytes por matriz e nível além dos planos (16 nos tipos D)")
    g.add_argument("--niveis", nargs="+", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--plano-out", help="grava o nível de cada matriz (JSON)")
    g.add_argument("--plano", help="usa este plano (JSON com 'niveis', de strata_dkl.py plano) no lugar do alocador por erro")
    args = ap.parse_args()
    erros(args) if args.cmd == "erros" else grava(args)


if __name__ == "__main__":
    main()
