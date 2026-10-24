"""Mede a volta ao BF16 decodificando enquanto baixa, numa rede simulada (servidor local com teto
de banda), contra baixar tudo e só depois juntar, e contra baixar o BF16 cru na mesma banda.

    python3 tests/dist/fluxo.py --dist <pasta com índice e estratos> --gguf <Q4_K_M.gguf> \\
        --bf16-bytes 2471645608 --mbit 100,300,1000,0 --saida resultados/fluxo

Quem roda já tem o GGUF; só os arquivos de estrato passam pela "rede".
"""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--dist", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--bf16-bytes", type=int, required=True, help="tamanho dos shards BF16 originais, para a comparação")
ap.add_argument("--mbit", default="100,300,1000,0", help="tetos de banda (0 = sem teto)")
ap.add_argument("--bin", default="leitor-rs/target/release/strata-ref")
ap.add_argument("--tmp", default="/tmp/strata_fluxo")
ap.add_argument("--saida", required=True)
ap.add_argument("--conexoes", type=int, default=8)
ap.add_argument("--porta", type=int, default=8765)
a = ap.parse_args()
AQUI = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.abspath(a.bin)
IDX = "model.strata.index.json"
os.makedirs(a.saida, exist_ok=True)
resto = sum(os.path.getsize(os.path.join(a.dist, f)) for f in os.listdir(a.dist) if f.endswith(".strata"))


def junta(pasta, extra=()):
    t = time.time()
    r = subprocess.run([BIN, "junta", os.path.join(pasta, IDX), "--base", a.gguf, *extra], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode or r.stdout.count("(confere)") == 0:
        sys.exit("junta falhou:\n" + r.stdout[-2000:])
    return time.time() - t, r.stdout


# a volta com tudo já no disco (o caso "baixa tudo e depois junta")
t_local, log = junta(a.dist)
print(f"junta com tudo no disco: {t_local:.0f} s", flush=True)
linhas = []
for mbit in [float(x) for x in a.mbit.split(",")]:
    srv = subprocess.Popen([sys.executable, os.path.join(AQUI, "servidor_faixas.py"), a.dist, "--porta", str(a.porta), "--mbit", str(mbit)])
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.porta}/{IDX}").read()
            break
        except Exception:
            time.sleep(0.1)
    shutil.rmtree(a.tmp, ignore_errors=True)
    t0 = time.time()
    bx = subprocess.Popen([sys.executable, os.path.join(AQUI, "baixa_estratos.py"), f"http://127.0.0.1:{a.porta}", a.tmp, "--conexoes", str(a.conexoes)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while not os.path.exists(os.path.join(a.tmp, IDX)):
        if bx.poll() is not None:
            sys.exit("baixador falhou:\n" + bx.stdout.read())
        time.sleep(0.05)
    tj, logj = junta(a.tmp, ["--espera"])
    t_total = time.time() - t0
    saida_bx = bx.communicate()[0]
    srv.terminate()
    srv.wait()
    m = re.search(r"baixado: (\d+) bytes em ([\d.]+) s", saida_bx)
    t_down = float(m.group(2))
    taxa = int(m.group(1)) / t_down  # bytes/s medidos
    t_bf16 = a.bf16_bytes / taxa
    r = {"mbit": mbit, "mbit_medido": 8 * taxa / 1e6, "baixar_estratos_s": t_down, "fluxo_total_s": t_total,
         "baixa_depois_junta_s": t_down + t_local, "bf16_cru_s": t_bf16, "tempos_junta": [l for l in logj.splitlines() if l.startswith("tempos:")]}
    linhas.append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
    shutil.rmtree(a.tmp, ignore_errors=True)

json.dump({"resto_bytes": resto, "bf16_bytes": a.bf16_bytes, "junta_local_s": t_local, "rodadas": linhas}, open(os.path.join(a.saida, "fluxo.json"), "w"), indent=1)
md = [f"# Volta ao BF16 decodificando enquanto baixa", "",
      f"Estratos: {resto / 1e9:.2f} GB · BF16: {a.bf16_bytes / 1e9:.2f} GB · `junta` com tudo no disco: {t_local:.0f} s · {os.cpu_count()} núcleos", "",
      "| rede (medida) | baixar os estratos | baixa e depois junta | **em fluxo** | baixar o BF16 cru |", "|---|---:|---:|---:|---:|"]
for r in linhas:
    rede = f"{r['mbit_medido']:.0f} Mbit/s" + ("" if r["mbit"] else " (sem teto)")
    md.append(f"| {rede} | {r['baixar_estratos_s']:.0f} s | {r['baixa_depois_junta_s']:.0f} s | **{r['fluxo_total_s']:.0f} s** | {r['bf16_cru_s']:.0f} s |")
open(os.path.join(a.saida, "fluxo.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
