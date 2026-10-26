"""Testa o GGUF como receita: requantiza o f16.gguf de um repositório do Hub com o imatrix publicado
e o llama-quantize da mesma versão do llama.cpp, e compara o SHA-256 de cada variante com o do Hub.

    python3 tests/dist/receita_gguf.py --repo bartowski/Llama-3.2-1B-Instruct-GGUF \\
        --f16 pub/Llama-3.2-1B-Instruct-f16.gguf \\
        --imatrix /models_out/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct.imatrix \\
        --quantize llama.cpp-b3821/build/bin/llama-quantize --saida resultados/receita_1b

O caminho do imatrix entra nos metadados do GGUF (quantize.imatrix.file); para o SHA bater, ele tem
de ser o mesmo que o autor usou (um link simbólico basta). Cada saída é apagada depois do hash.
"""
import argparse, hashlib, json, os, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--f16", required=True)
ap.add_argument("--imatrix", required=True)
ap.add_argument("--quantize", required=True)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("--variantes", default="", help="lista separada por vírgula; vazio = todas do repositório")
ap.add_argument("--tmp", default="/tmp/strata_receita")
ap.add_argument("--saida", required=True)
a = ap.parse_args()
os.makedirs(a.saida, exist_ok=True)
os.makedirs(a.tmp, exist_ok=True)

arvore = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{a.repo}/tree/main"))
hub = {f["path"]: (f["size"], f["lfs"]["oid"]) for f in arvore if f["path"].endswith(".gguf") and f.get("lfs")}
f16_nome = os.path.basename(a.f16)
prefixo = f16_nome[: -len("-f16.gguf")]


def receitas(v):
    """Comandos candidatos para a variante v, do mais provável ao menos."""
    base, extra = v, []
    if v.endswith("_XL"):  # Q3_K_XL = Q3_K_L com embeddings e saída em Q8_0
        base, extra = v[:-2] + "L", ["--output-tensor-type", "Q8_0", "--token-embedding-type", "Q8_0"]
    elif v.endswith("_L") and v not in ("Q3_K_L",):
        base = "Q6_K" if v == "Q6_K_L" else v[:-2] + "_M"
        extra = ["--output-tensor-type", "Q8_0", "--token-embedding-type", "Q8_0"]
    yield "imatrix", ["--imatrix", a.imatrix, *extra], base
    yield "sem imatrix", extra, base


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


nomes = sorted(n for n in hub if n != f16_nome)
if a.variantes:
    nomes = [f"{prefixo}-{v}.gguf" for v in a.variantes.split(",")]
linhas = []
for n in nomes:
    v = n[len(prefixo) + 1 : -len(".gguf")]
    tam, oid = hub[n]
    r = {"variante": v, "bytes": tam, "sha_hub": oid, "bate": False, "tentativas": []}
    for rotulo, extra, base in receitas(v):
        out = os.path.join(a.tmp, n)
        cmd = [a.quantize, *extra, a.f16, out, base, str(a.threads)]
        t = time.time()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t
        if p.returncode:
            r["tentativas"].append({"receita": rotulo, "erro": p.stdout[-500:]})
            continue
        s, b = sha(out), os.path.getsize(out)
        os.remove(out)
        r["tentativas"].append({"receita": rotulo, "cmd": " ".join(cmd[1:]), "s": round(dt, 1), "sha": s, "bytes": b})
        if s == oid:
            r.update(bate=True, receita=rotulo, s=round(dt, 1))
            break
    print(f"{v:10} {'bate' if r['bate'] else 'NÃO bate'} {r.get('receita', '')} {r.get('s', '')}", flush=True)
    linhas.append(r)

json.dump(linhas, open(os.path.join(a.saida, "receita.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(a.saida, "receita.md"), "w") as f:
    f.write(f"# Receita: {a.repo}\n\nRequantizado de `{f16_nome}` com `{os.path.basename(a.quantize)}`, {a.threads} threads.\n\n")
    f.write("| Variante | Bytes | SHA-256 igual ao do Hub | Receita | Tempo (s) |\n|---|---|---|---|---|\n")
    for r in linhas:
        f.write(f"| {r['variante']} | {r['bytes']:,} | {'sim' if r['bate'] else '**não**'} | {r.get('receita', '-')} | {r.get('s', '-')} |\n")
    ok = sum(r["bate"] for r in linhas)
    tot = sum(r["bytes"] for r in linhas)
    f.write(f"\n{ok}/{len(linhas)} variantes batem; {tot / 1e9:.2f} GB de GGUF quantizado no repositório.\n")
print(f"{sum(r['bate'] for r in linhas)}/{len(linhas)} batem")
sys.exit(0 if all(r["bate"] for r in linhas) else 1)
