"""Teste completo de armazenamento do STRATA sobre o GGUF do Hub (docs/02 B.10) num modelo grande.

Pensado para o Colab (CPU com muita RAM), mas roda em qualquer Linux com disco suficiente:
  1. baixa o GGUF publicado e os shards BF16, um shard por vez e no máximo `--adiante` à frente;
  2. grava os arquivos de estrato com `strata-ref distribui --espera --apaga-shards` (cada shard é
     apagado depois de lido, então o disco precisa de GGUF + resto + uns poucos shards);
  3. confere a volta bit a bit com `strata-ref junta` e compara o SHA-256 de cada shard
     remontado com o oid LFS do Hugging Face;
  4. faz a conta de armazenamento com os tamanhos reais do Hub (API /tree).

    python3 tests/dist/teste_grande.py --hf unsloth/Meta-Llama-3.1-70B-Instruct \\
        --gguf-repo bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \\
        --gguf Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf --dir /content/m70

Saída: <dir>/resultados/{relatorio.md, resultado.json, distribui.txt, junta.txt}.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--hf", required=True, help="repositório com os .safetensors BF16")
ap.add_argument("--gguf-repo", required=True)
ap.add_argument("--gguf", required=True, help="nome do arquivo Q4_K_M no repositório de GGUF")
ap.add_argument("--dir", required=True)
ap.add_argument("--bin", default="leitor-rs/target/release/strata-ref")
ap.add_argument("--niveis", type=int, default=3)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--adiante", type=int, default=2, help="shards baixados à frente do escritor")
ap.add_argument("--forca", action="store_true", help="roda mesmo se o disco parecer pequeno")
ap.add_argument("--tudo", action="store_true", help="na volta, confere também o SHA-256 de cada arquivo inteiro (lê tudo duas vezes)")
a = ap.parse_args()

D = os.path.abspath(a.dir)
SAI = os.path.join(D, "dist")
RES = os.path.join(D, "resultados")
os.makedirs(SAI, exist_ok=True)
os.makedirs(RES, exist_ok=True)
BIN = os.path.abspath(a.bin)
G = lambda x: f"{x / 1e9:.2f} GB"


def arvore(repo):
    """Arquivos do repositório (nome, bytes, oid LFS), pela API pública do Hub."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    req = urllib.request.Request(url)
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.urlopen(req) as r:
        d = json.load(r)
    return {f["path"]: (f.get("size", 0), (f.get("lfs") or {}).get("oid")) for f in d if f.get("type") == "file"}


from huggingface_hub import hf_hub_download  # noqa: E402

hf = arvore(a.hf)
gg = arvore(a.gguf_repo)
if a.gguf not in gg:
    sys.exit(f"{a.gguf} não está em {a.gguf_repo}")
idx_path = hf_hub_download(a.hf, "model.safetensors.index.json", local_dir=D)
shards = sorted(set(json.load(open(idx_path))["weight_map"].values()))
bf16 = sum(hf[s][0] for s in shards)
gtam = gg[a.gguf][0]
maior = max(hf[s][0] for s in shards)
resto_est = 0.40 * bf16  # ~6,2 bits/peso de 16 (seção 31), com folga
precisa = gtam + resto_est + (a.adiante + 1) * maior + 5e9
livre = shutil.disk_usage(D).free
print(f"{a.hf}: {len(shards)} shards, {G(bf16)}; GGUF {G(gtam)}")
print(f"disco: livre {G(livre)}, precisa ~{G(precisa)}")
if livre < precisa and not a.forca:
    sys.exit("disco pequeno demais: use um ambiente com mais disco, um modelo menor ou --forca")

t0 = time.time()
print("baixando o GGUF...", flush=True)
gpath = hf_hub_download(a.gguf_repo, a.gguf, local_dir=D)
t_gguf = time.time() - t0
print(f"GGUF em {t_gguf:.0f} s", flush=True)

# baixador: um shard por vez, no máximo `adiante` no disco esperando o escritor
erro = []
baixados = []


def baixa():
    try:
        for s in shards:
            while sum(os.path.exists(os.path.join(D, x)) for x in shards) >= a.adiante:
                time.sleep(1)
            t = time.time()
            hf_hub_download(a.hf, s, local_dir=D)  # grava em .incomplete e renomeia no fim
            baixados.append((s, time.time() - t))
            print(f"  baixado {s} ({len(baixados)}/{len(shards)}) em {time.time() - t:.0f} s", flush=True)
    except Exception as e:  # o escritor espera para sempre; quem roda o processo o mata
        erro.append(repr(e))


th = threading.Thread(target=baixa, daemon=True)
th.start()

try:
    import psutil
except ImportError:
    psutil = None


def roda(cmd, log):
    """Roda e registra a saída e o pico de RAM (RSS) do processo."""
    t = time.time()
    pico = 0
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        ps = psutil.Process(p.pid) if psutil else None
        def mede():
            nonlocal pico
            while p.poll() is None:
                try:
                    pico = max(pico, ps.memory_info().rss)
                except Exception:
                    pass
                time.sleep(1)
        if ps:
            threading.Thread(target=mede, daemon=True).start()
        for linha in p.stdout:
            f.write(linha)
            print("  " + linha, end="", flush=True)
            if erro:
                p.kill()
                break
        p.wait()
    if erro:
        sys.exit("download falhou: " + erro[0])
    if p.returncode != 0:
        sys.exit(f"{cmd[1]} falhou (código {p.returncode}); ver {log}")
    return time.time() - t, pico


thr = ["--threads", str(a.threads)] if a.threads else []
caminhos = [os.path.join(D, s) for s in shards]
print("gravando os arquivos de estrato (cada shard é apagado depois de lido)...", flush=True)
t_esc, ram_esc = roda([BIN, "distribui", gpath, SAI, *caminhos, "--index", idx_path, "--niveis", str(a.niveis), "--espera", "--apaga-shards", *thr],
                      os.path.join(RES, "distribui.txt"))
th.join()
print(f"gravado em {t_esc:.0f} s", flush=True)

print("conferindo a volta bit a bit (sem gravar os shards)...", flush=True)
t_vol, ram_vol = roda([BIN, "junta", os.path.join(SAI, "model.strata.index.json"), "--base", gpath, *(["--tudo"] if a.tudo else []), *thr], os.path.join(RES, "junta.txt"))

# SHA-256 remontado contra o oid LFS do Hub
rem = dict(re.findall(r"^(\S+\.safetensors): \d+ bytes, sha256 ([0-9a-f]{64}) \(confere\)", open(os.path.join(RES, "junta.txt")).read(), re.M))
iguais = [s for s in shards if rem.get(s) and rem[s] == hf[s][1]]
print(f"SHA-256 igual ao do Hugging Face: {len(iguais)}/{len(shards)} shards", flush=True)

# tamanhos
rel = open(os.path.join(RES, "distribui.txt")).read()
arquivos = {m.group(2).split()[-1]: int(m.group(1)) for m in re.finditer(r"^\s*(\d+) bytes\s+(.*)$", rel, re.M)}
resto = sum(v for k, v in arquivos.items() if not k.endswith(".gguf"))
pesos = int(re.search(r"total:\s+(\d+) M pesos", rel).group(1)) * 1e6
somas = {m.group(1): m.group(0).strip() for m in re.finditer(r"^\s*(\S+): .* bits/peso .*$", rel, re.M)}


def variante(v):
    """Bytes de uma variante no repositório de GGUF (arquivo único ou dividido em partes)."""
    return sum(sz for p, (sz, _) in gg.items() if p.endswith(".gguf") and re.search(rf"-{re.escape(v)}(\.|-\d{{5}}-of-|/)", p))


q = {v: variante(v) for v in ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]}
todos = sum(sz for p, (sz, _) in gg.items() if p.endswith(".gguf"))
nvar = len({re.sub(r"-\d{5}-of-\d{5}", "", p.split("/")[0]) for p in gg if p.endswith(".gguf")})
tudo = gtam + resto
sep = bf16 + q["Q4_K_M"] + q["Q5_K_M"] + q["Q6_K"]
r = {
    "modelo": a.hf, "gguf": f"{a.gguf_repo}/{a.gguf}", "niveis": a.niveis, "pesos_matrizes": pesos,
    "bytes": arquivos, "resto": resto, "total": tudo, "bits_por_peso": 8 * tudo / pesos, "resto_bits_por_peso": 8 * resto / pesos,
    "somas": somas, "bf16": bf16, "hub": q, "repo_gguf_total": todos, "repo_gguf_variantes": nvar,
    "sha_iguais": len(iguais), "shards": len(shards),
    "tempos_s": {"gguf": t_gguf, "shards": sum(t for _, t in baixados), "grava": t_esc, "volta": t_vol},
    "ram_pico_gb": {"grava": ram_esc / 1e9, "volta": ram_vol / 1e9}, "nucleos": os.cpu_count(),
}
json.dump(r, open(os.path.join(RES, "resultado.json"), "w"), indent=1)

md = [f"# Teste de armazenamento: {a.hf}", "",
      f"GGUF base: `{a.gguf_repo}/{a.gguf}` (sem mudança) · K = {a.niveis} · {os.cpu_count()} núcleos", "",
      "| Arquivo | bytes |", "|---|---:|"]
md += [f"| `{k}` | {v:,} |".replace(",", ".") for k, v in arquivos.items()]
md += ["", "```", *somas.values(), "```", "",
       f"- **Total:** {G(tudo)} = {8 * tudo / pesos:.3f} bits/peso ({pesos / 1e9:.1f} G pesos em matrizes)",
       f"- **Volta:** {len(iguais)}/{len(shards)} shards com o SHA-256 do Hugging Face",
       f"- **Hub hoje (BF16 + Q4_K_M + Q5_K_M + Q6_K):** {G(sep)} → {G(tudo)} ({1 - tudo / sep:.1%} menos)",
       f"- **Quem já tem o Q4_K_M e quer o BF16 exato:** baixa {G(resto)} em vez de {G(bf16)} ({1 - resto / bf16:.1%} menos)",
       f"- **Repositório de GGUF:** {nvar} variantes, {G(todos)}; as 3 cobertas são {(q['Q4_K_M'] + q['Q5_K_M'] + q['Q6_K']) / todos:.0%}",
       f"- **Tempos:** GGUF {t_gguf:.0f} s, shards {sum(t for _, t in baixados):.0f} s (download), gravar {t_esc:.0f} s, voltar {t_vol:.0f} s",
       f"- **Pico de RAM:** gravar {ram_esc / 1e9:.1f} GB, voltar {ram_vol / 1e9:.1f} GB",
       "", "Q5_K_M " + G(q["Q5_K_M"]) + " · Q6_K " + G(q["Q6_K"]) + " · Q8_0 " + G(q["Q8_0"])]
open(os.path.join(RES, "relatorio.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
if len(iguais) != len(shards):
    sys.exit("ATENÇÃO: algum shard não voltou com o SHA-256 do Hugging Face")
