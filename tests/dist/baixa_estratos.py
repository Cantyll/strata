"""Baixa os arquivos de estrato de um índice B.10 na ordem em que o `strata-ref junta --espera` os
lê, para a volta ao BF16 começar enquanto o download ainda corre.

Cada arquivo de estrato é lido pelo `junta` do começo ao fim, tensor a tensor, e os tensores se
alternam entre os arquivos. O baixador corta essa sequência em pedaços (faixas HTTP), pede vários
ao mesmo tempo e grava cada arquivo em ordem, sempre no fim dele: o tamanho do arquivo é o quanto
já chegou, que é o que o `junta --espera` olha. O GGUF não é baixado (quem usa isto já o tem).

    python3 tests/dist/baixa_estratos.py <url da pasta> <pasta local> [--conexoes 8] [--pedaco-mb 32]
"""
import argparse, base64, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url", help="pasta remota com model.strata.index.json e os arquivos de estrato")
ap.add_argument("pasta")
ap.add_argument("--conexoes", type=int, default=8)
ap.add_argument("--pedaco-mb", type=float, default=32)
ap.add_argument("--indice", default="model.strata.index.json")
a = ap.parse_args()
os.makedirs(a.pasta, exist_ok=True)
URL = a.url.rstrip("/") + "/"
CAB = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}


def pega(nome, ini=None, fim=None):
    req = urllib.request.Request(URL + nome, headers=dict(CAB))
    if ini is not None:
        req.add_header("Range", f"bytes={ini}-{fim - 1}")
    for tentativa in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if ini is not None and len(b) != fim - ini:
                raise IOError(f"{nome}: faixa {ini}-{fim} veio com {len(b)} bytes")
            return b
        except Exception:
            if tentativa == 4:
                raise
            time.sleep(2 ** tentativa)


t0 = time.time()
ib = pega(a.indice)
tmp = os.path.join(a.pasta, a.indice + ".parcial")
open(tmp, "wb").write(ib)
os.replace(tmp, os.path.join(a.pasta, a.indice))  # o junta só vê o índice inteiro
idx = json.loads(ib)

# a ordem de leitura do junta: shards na ordem do índice, tensores na ordem dos offsets do shard
onde = {(t["shard"], t["name"]): i for i, t in enumerate(idx["tensors"])}
por = {}
for s in idx["streams"]:
    por.setdefault(s["t"], []).append(s)
arqs = idx["arquivos"]
pos = {f: 16 for f in range(1, len(arqs))}
fila = [(f, 0, 16) for f in range(1, len(arqs))]  # os preâmbulos primeiro
aberto = {}
lim = int(a.pedaco_mb * 2 ** 20)
for si, fo in enumerate(idx["fonte"]["shards"]):
    h = json.loads(base64.b64decode(fo["header_b64"]))
    for _, nome in sorted((v["data_offsets"][0], k) for k, v in h.items() if k != "__metadata__"):
        for s in sorted(por.get(onde[(si, nome)], []), key=lambda s: s["s"]):
            f = s["f"]
            if f == 0:
                continue
            if s["o"] != pos[f]:
                sys.exit(f"{arqs[f]['nome']}: stream fora de ordem no índice; o baixador em fluxo precisa da ordem do escritor")
            pos[f] = s["o"] + s["l"]
            aberto.setdefault(f, s["o"])
            # quando o que falta pedir passa de um pedaço, pede um pouco de cada arquivo: todos
            # avançam no ritmo em que o junta os consome (o e00, das normas, não fica para o fim)
            if sum(pos[g] - i for g, i in aberto.items()) >= lim:
                fila += [(g, i, pos[g]) for g, i in sorted(aberto.items()) if pos[g] > i]
                aberto = {}
fila += [(g, i, pos[g]) for g, i in sorted(aberto.items()) if pos[g] > i]
for f in pos:
    if pos[f] != arqs[f]["bytes"]:
        sys.exit(f"{arqs[f]['nome']}: os streams não cobrem o arquivo")

saidas = {f: open(os.path.join(a.pasta, arqs[f]["nome"]), "wb") for f in range(1, len(arqs))}
total = sum(fim - ini for _, ini, fim in fila)
feito = 0
with ThreadPoolExecutor(a.conexoes) as ex:
    janela = []
    prox = 0
    while prox < len(fila) or janela:
        while prox < len(fila) and len(janela) < 2 * a.conexoes:
            f, ini, fim = fila[prox]
            janela.append((f, ex.submit(pega, arqs[f]["nome"], ini, fim)))
            prox += 1
        f, fut = janela.pop(0)
        b = fut.result()
        saidas[f].write(b)
        saidas[f].flush()
        feito += len(b)
for s in saidas.values():
    s.close()
dt = time.time() - t0
print(f"baixado: {feito} bytes em {dt:.1f} s ({8 * feito / dt / 1e6:.0f} Mbit/s), {len(fila)} faixas")
