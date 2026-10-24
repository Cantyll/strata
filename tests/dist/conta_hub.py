"""Quanto a distribuição sobre GGUF (docs/02 B.10) ocupa contra os arquivos que o Hub guarda hoje.

Lê as listagens do Hub salvas em resultados/ (API /tree/main) e o relatório do `strata-ref distribui`.
    python3 tests/dist/conta_hub.py
"""
import json, os, re

R = os.path.join(os.path.dirname(__file__), "resultados")
gguf = {f["path"]: f.get("size", 0) for f in json.load(open(f"{R}/hub_bartowski_Meta-Llama-3.1-8B-Instruct-GGUF.json"))}
hf = {f["path"]: f.get("size", 0) for f in json.load(open(f"{R}/hub_unsloth_Meta-Llama-3.1-8B-Instruct.json"))}
rel = open(f"{R}/Meta-Llama-3.1-8B-Instruct_distribui.txt").read()

P = "Meta-Llama-3.1-8B-Instruct-"
q = {k: gguf[P + k + ".gguf"] for k in ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]}
bf16 = sum(v for k, v in hf.items() if k.endswith(".safetensors"))
arquivos = {}
for m in re.finditer(r"^\s*(\d+) bytes\s+(.*)$", rel, re.M):
    arquivos[m.group(2).split()[-1]] = int(m.group(1))
resto = sum(v for k, v in arquivos.items() if not k.endswith(".gguf"))
pesos = int(re.search(r"total:\s+(\d+) M pesos", rel).group(1)) * 1e6
# STRATA 0.2 sem perdas medido no Llama-3.1-8B base (seção 19): 10,57 bits/peso -> estimativa
z02 = 10.57 / 16 * bf16
G = lambda x: f"{x / 1e9:6.2f} GB"

print(f"BF16 safetensors          {G(bf16)}")
for k, v in q.items():
    print(f"{k:<25} {G(v)}")
print(f"resto (níveis+fecho+índice) {G(resto)}   ({8 * resto / pesos:.2f} bits/peso das matrizes)")
tudo = q["Q4_K_M"] + resto
print(f"GGUF + resto              {G(tudo)}   ({8 * tudo / pesos:.2f} bits/peso)")
sep = bf16 + q["Q4_K_M"] + q["Q5_K_M"] + q["Q6_K"]
sep02 = z02 + q["Q4_K_M"] + q["Q5_K_M"] + q["Q6_K"]
print(f"\nguardados separados (BF16 cru + Q4_K_M + Q5_K_M + Q6_K) {G(sep)} -> {1 - tudo / sep:.1%} menos")
print(f"idem com o BF16 em STRATA 0.2 (estimado {G(z02)})     {G(sep02)} -> {1 - tudo / sep02:.1%} menos")
print(f"\nquem tem o Q4_K_M e quer o BF16 exato: baixa {G(resto)} em vez de {G(bf16)} ({1 - resto / bf16:.1%} menos)")
print(f"  ou do .strata 0.2 sozinho {G(z02)} ({1 - resto / z02:.1%} menos)")
print(f"quem tem o Q4_K_M e quer o nível 1: baixa {G(arquivos['model-e01-00000-of-00001.strata'])} em vez do Q5_K_M {G(q['Q5_K_M'])}")
todos = sum(v for k, v in gguf.items() if k.endswith(".gguf"))
print(f"\nrepositório de GGUF inteiro: {sum(k.endswith('.gguf') for k in gguf)} arquivos, {G(todos)}; os 3 que o arquivo cobre (Q4_K_M, Q5_K_M, Q6_K) são {(q['Q4_K_M'] + q['Q5_K_M'] + q['Q6_K']) / todos:.0%}")
