#!/usr/bin/env python3
"""Resume o CSV de strata_subir gerar: tempo por token antes, durante e depois da subida de nível.

    python tests/demanda/resumo_subida.py saida.csv [...] [--janela INI FIM]

Com --janela, o "durante" é a mesma faixa de tokens em todos os arquivos: é como comparar com um controle sem subida
(--alvo 0), que não tem trocas para marcar a faixa.

"Pausa" = quanto um token demorou além do normal. Normal = mediana dos tokens antes da subida (nível 0, mesmas
threads). Os tokens do fim ficam mais lentos também porque o contexto cresce; por isso o "depois" é só indicativo."""
import csv, statistics as st, sys

args = sys.argv[1:]
janela = None
if "--janela" in args:
    j = args.index("--janela")
    janela = (int(args[j + 1]), int(args[j + 2]))
    del args[j:j + 3]
for arq in args:
    r = list(csv.DictReader(open(arq)))
    t = [1000 * float(x["segundos"]) for x in r]
    n1 = [int(x["nivel1_matrizes"]) for x in r]
    n2 = [int(x["nivel2_matrizes"]) for x in r]
    trocas = [int(x["trocas"]) for x in r]
    tot = max(n1)
    if janela:
        ini, fim = janela[0] + 1, janela[1]
    elif any(trocas):
        ini, fim = next(i for i, x in enumerate(trocas) if x), max(i for i, x in enumerate(trocas) if x)
    else:
        sys.exit(f"{arq}: sem trocas; passe --janela INI FIM")
    antes, dur = t[:ini - 1], t[ini - 1:fim + 1]
    base = st.median(antes)
    dur_s = sorted(dur)
    print(f"{arq}: antes p50 {base:.1f} ms | durante {len(dur)} tokens, p50 {st.median(dur):.1f}, p99 {dur_s[max(0, int(.99 * len(dur)) - 1)]:.1f}, "
          f"máx {max(dur):.1f} ms (máx = {max(dur) / base:.2f} tokens normais; pausa = {max(dur) / base - 1:.2f}) | "
          f"depois p50 {st.median(t[fim + 1:len(t) - 8]) if len(t) - 8 > fim + 1 else float('nan'):.1f} | trocas aplicadas {sum(trocas)} em {sum(1 for x in trocas if x)} tokens, "
          f"máx {max(float(x['troca_us']) for x in r):.0f} µs | níveis completos: 1 = {bool(tot) and n1[-1] == tot}, 2 = {bool(tot) and n2[-1] == tot}")
    if "t" in r[0] and tot:                                           # E4: quando cada nível ficou completo, desde o início da geração
        tt = [float(x["t"]) for x in r]
        q = lambda n: next((tt[i] for i in range(len(r)) if n[i] == tot), float("nan"))
        print(f"    nível 1 completo em {q(n1):.1f} s, nível 2 em {q(n2):.1f} s; tokens/s durante a subida "
              f"{len(dur) / sum(dur) * 1000:.1f}, antes {1000 / base:.1f}")
