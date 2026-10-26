#!/usr/bin/env python3
"""Gera tests/kernel/strata_8b_gptq_colab.ipynb (C1 de docs/10 no Llama-3.1-8B, Colab A100).

    python tests/kernel/gera_nb_gptq.py SHA_DO_STRATA
"""
import json, os, sys

SHA = sys.argv[1]
AQUI = os.path.dirname(os.path.abspath(__file__))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


celulas = [
    md("""
# STRATA, C1: bits dos níveis escolhidos com a Hessiana (Llama-3.1-8B)

**Antes de rodar:** *Ambiente de execução → Alterar o tipo de ambiente de execução* → **A100** com **RAM alta**. Depois *Executar tudo*. Leva umas 3 a 4 horas e precisa de ~150 GB de disco.

O plano está em `docs/10-motor-de-compressao.md`. O nível 0 é o Q4_K_M com imatrix, intacto byte a byte. Os pontos dos filhos de cada nível são os dos tipos de hoje:
- **P**: o meio de cada metade da célula; é o tipo que passa na velocidade;
- **D**: deslocamentos por código, por tensor.

Muda só **qual filho cada peso usa**. Hoje é o mais próximo do peso original. Com `gptq`, o escritor percorre as entradas de cada matriz, escolhe o filho mais próximo do alvo já compensado e passa o erro para as entradas que faltam, pela Hessiana das ativações (H = XᵀX, a matriz inteira da qual a imatrix é a diagonal). O formato e o kernel não mudam: os planos continuam com 1 bit por peso.

**No Llama-3.2-1B** (contêiner, caminho F16, 40 trechos, KL ×1000): P de hoje 15,82 / 4,39; P com GPTQ **8,68 / 3,71**; Q5_K_M 11,79, Q6_K 5,45 (níveis 1 / 2). O 1B já prometeu mais do que o 8B entregou pelo menos quatro vezes. Por isso a decisão é deste notebook.

**Caminho de medida:** tudo em F16 (pesos desquantizados), STRATA e k-quants, para que nenhum lado arredonde as ativações. A KL é medida contra o BF16 em dois textos:
- 100 trechos do wikitext-2 de teste (a calibração usa o de treino);
- 100 trechos do C4, fora do domínio da calibração.

**Régua (escrita antes de rodar, docs/10):** bits por peso a mais que a frente dos k-quants com imatrix em F16 (Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0), no mesmo tamanho de RAM. O nível 1 é contado com o Q6_K empacotado.
- **Sucesso:** P com GPTQ ≤ +0,05 bit nos níveis 1 e 2 no wikitext, e no C4 o ganho sobre o P de hoje mantém pelo menos metade (em log da KL).
- **Zona cinza:** algum nível entre +0,05 e +0,12, ou só os D chegam lá (D reprova na velocidade).
- **Fracasso:** nível 1 acima de +0,12 com P e com D.
"""),
    code("""
KLD_CHUNKS = 100           # trechos na KL, em cada texto (~65 MB de disco por trecho)
IMATRIX_CHUNKS = 125       # trechos de 512 tokens do wikitext de treino, para a imatrix e para as Hessianas
GRUPO = 8                  # camadas por passada na coleta das Hessianas (8 cabem numa A100 de 40 GB)
MEDIR_D_PERTO = False      # D sem GPTQ: já medido no 8B (seção 28: 9,03 / 3,35); True mede de novo (+30 min)
MODELO_HF = "unsloth/Meta-Llama-3.1-8B"
"""),
    code("""
import os, re, json, shutil, subprocess, time
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
assert gpu.returncode == 0, 'Sem GPU NVIDIA: troque o tipo de ambiente de execução para A100.'
GPU = gpu.stdout.strip().splitlines()[0]
N = os.cpu_count()
RAM = int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1]) / 1e6
DISCO = shutil.disk_usage('/content').free / 1e9
print(f'GPU: {GPU} | CPU: {N} threads | RAM {RAM:.0f} GB | disco livre {DISCO:.0f} GB')
if DISCO < 150:
    print('Aviso: pouco disco. Ponha KLD_CHUNKS = 50.')
if RAM < 50:
    print('Aviso: pouca RAM. Ligue a RAM alta.')
T0 = time.time()
"""),
    code(f"""
STRATA_SHA = "{SHA}"   # o commit do STRATA com o GPTQ dos níveis (o branch continua mudando)
LLAMA_SHA = "2a667032dc788c3a071d538e0b572e45a3f12731"
!rm -rf strata && git init -q strata && cd strata && git remote add origin https://github.com/Cantyll/strata.git && git fetch -q --depth 1 origin $STRATA_SHA && git checkout -q FETCH_HEAD
!rm -rf llama.cpp && git init -q llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch -q --depth 1 origin $LLAMA_SHA && git checkout -q FETCH_HEAD
!pip install -q -U transformers safetensors
!pip install -q -e llama.cpp/gguf-py
# só o caminho F16: o llama.cpp não precisa do patch dos tipos
!cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DGGML_NATIVE=ON 2>&1 | tail -3 && cmake --build build -j $(nproc) --target llama-quantize llama-perplexity llama-imatrix 2>&1 | tail -5
!ls -la llama.cpp/build/bin | grep -E 'llama-(perplexity|quantize|imatrix)$'
"""),
    code("""
import urllib.request, zipfile, io, gzip
from huggingface_hub import hf_hub_download
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip').read()))
open('/content/wiki.test.raw', 'wb').write(z.read('wikitext-2-raw/wiki.test.raw'))
open('/content/wiki.train.raw', 'wb').write(z.read('wikitext-2-raw/wiki.train.raw'))
# segundo texto: o começo do primeiro arquivo de validação do C4 (inglês), ~1,5 MB, fora do domínio da calibração
p = hf_hub_download('allenai/c4', 'en/c4-validation.00000-of-00008.json.gz', repo_type='dataset')
txt, n = [], 0
with gzip.open(p, 'rt') as f:
    for l in f:
        t = json.loads(l)['text']; txt.append(t); n += len(t)
        if n > 1_500_000:
            break
open('/content/c4.test.raw', 'w').write('\\n\\n'.join(txt))
for f in ['wiki.test.raw', 'wiki.train.raw', 'c4.test.raw']:
    print(f, os.path.getsize('/content/' + f), 'bytes')
"""),
    md("Modelo em BF16, imatrix, k-quants com imatrix e as duas bases da KL. Os pesos no formato do Hugging Face ficam até as Hessianas serem coletadas."),
    code("""
from huggingface_hub import snapshot_download
src = snapshot_download(MODELO_HF, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
G = '/content/gguf'
os.makedirs(G, exist_ok=True)
B = 'llama.cpp/build/bin'
!python llama.cpp/convert_hf_to_gguf.py {src} --outtype bf16 --outfile {G}/BF16.gguf 2>&1 | tail -1
!{B}/llama-imatrix -m {G}/BF16.gguf -f /content/wiki.train.raw -ngl 99 -c 512 --chunks {IMATRIX_CHUNKS} -o {G}/imatrix.gguf > {G}/imatrix.log 2>&1
!tail -2 {G}/imatrix.log
for q in ['Q4_K_M', 'Q5_K_S', 'Q5_K_M', 'Q6_K']:
    !{B}/llama-quantize --imatrix {G}/imatrix.gguf {G}/BF16.gguf {G}/{q}_imat.gguf {q} > {G}/{q}_imat.log 2>&1
!{B}/llama-quantize {G}/BF16.gguf {G}/Q8_0.gguf Q8_0 > {G}/Q8_0.log 2>&1
BASE = {'wiki': '/content/kld_wiki.bin', 'c4': '/content/kld_c4.bin'}
!{B}/llama-perplexity -m {G}/BF16.gguf -f /content/wiki.test.raw -c 512 -ngl 99 --chunks {KLD_CHUNKS} --kl-divergence-base {BASE['wiki']} > /dev/null 2>&1
!{B}/llama-perplexity -m {G}/BF16.gguf -f /content/c4.test.raw -c 512 -ngl 99 --chunks {KLD_CHUNKS} --kl-divergence-base {BASE['c4']} > /dev/null 2>&1
!ls -la {G}/*.gguf /content/kld_*.bin
!df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Hessianas das entradas de cada matriz, com os mesmos trechos da imatrix (≈ 1 GB por camada, em disco)."),
    code("""
!python strata/tests/kernel/strata_hessiana.py --modelo {src} --texto /content/wiki.train.raw --out /content/hess --chunks {IMATRIX_CHUNKS} --device cuda --grupo {GRUPO} 2>&1 | grep -v Warning | tail -3
!rm -rf {src}
!du -sh /content/hess; df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    code("""
kld = {}
def kl_de(path, nome):
    for txt, base in BASE.items():
        out = !{B}/llama-perplexity -m {path} --kl-divergence-base {base} --kl-divergence -ngl 99 2>&1
        s = '\\n'.join(out)
        k_ = re.search(r'Mean +KLD: +([0-9.]+) ± +([0-9.]+)', s)
        top = re.search(r'Same top p: +([0-9.]+) ± +([0-9.]+) %', s)
        if not k_:
            print(f'Erro na KL de {nome} ({txt}):', '\\n'.join(out[-8:]))
            continue
        kld[(nome, txt)] = (float(k_.group(1)), float(k_.group(2)), float(top.group(1)))
    print(nome + ': ' + ', '.join(f'{t} KL ×1000 {1000 * kld[(nome, t)][0]:.2f}' for t in BASE if (nome, t) in kld))
"""),
    md("Os k-quants no caminho F16 (a curva da régua)."),
    code("""
import gguf
tam = {}
ESC = f'python strata/tests/kernel/strata_escala_nivel.py --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf'
for q in ['Q4_K_M_imat', 'Q5_K_S_imat', 'Q5_K_M_imat', 'Q6_K_imat', 'Q8_0']:
    tam[q] = os.path.getsize(f'{G}/{q}.gguf') / 2**30
    !{ESC} --gguf {G}/{q}.gguf --modo copia --out /content/e.gguf 2>&1 | grep -v Warning | tail -1
    kl_de('/content/e.gguf', q)
    !rm -f /content/e.gguf
    if q != 'Q4_K_M_imat':
        !rm -f {G}/{q}.gguf
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("O STRATA: níveis 1 e 2 sobre o Q4_K_M com imatrix, pontos P e D, filho mais próximo (o de hoje) e GPTQ."),
    code("""
GQ = f'python strata/tests/kernel/strata_gptq_niveis.py --bf16 {G}/BF16.gguf --gguf {G}/Q4_K_M_imat.gguf --imatrix {G}/imatrix.gguf --hess /content/hess --device cuda'
os.makedirs('/content/res', exist_ok=True)
erro = {}
casos = [('P', 'perto'), ('P', 'gptq'), ('D', 'gptq')] + ([('D', 'perto')] if MEDIR_D_PERTO else [])
for pt, es in casos:
    n = f'{pt}_{es}'
    t = time.time()
    out = !{GQ} --pontos {pt} --escolha {es} --out /content/n1.gguf /content/n2.gguf --relatorio /content/res/erro_{n}.json 2>&1
    print(out[-1], f'({(time.time() - t) / 60:.0f} min)')
    m = re.search(r'nível 1 ([0-9.]+) nível 2 ([0-9.]+)', out[-1])
    if m:
        erro[n] = (float(m.group(1)), float(m.group(2)))
    for k in (1, 2):
        kl_de(f'/content/n{k}.gguf', f'{n}_N{k}')
    !rm -f /content/n1.gguf /content/n2.gguf
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("""
**A régua.** RAM do nível k = Q4_K_M + k bits por peso nas matrizes das camadas (o nível 1 com o Q6_K empacotado, como na seção 30). *bits a mais* = quanto o nível gasta além do k-quant em F16 de mesma KL (interpolação do log da KL entre os k-quants vizinhos). Abaixo de 0 = o STRATA ganha.
"""),
    code("""
import math
from IPython.display import Markdown, display
r = gguf.GGUFReader(f'{G}/Q4_K_M_imat.gguf')
n_cam = sum(int(t.n_elements) for t in r.tensors if t.name.startswith('blk.') and t.tensor_type.name in ('Q4_K', 'Q6_K') and t.n_elements > 1e6)
KQ = ['Q4_K_M_imat', 'Q5_K_S_imat', 'Q5_K_M_imat', 'Q6_K_imat', 'Q8_0']
RAM_N = {k: tam['Q4_K_M_imat'] + k * n_cam / 8 / 2**30 for k in (1, 2)}

def frente(txt):
    # pontos (GiB, KL) da frente: k-quants que nenhum outro domina (menor e com KL menor)
    pts = sorted((tam[q], kld[(q, txt)][0]) for q in KQ if (q, txt) in kld)
    return [p for p in pts if not any(o[0] <= p[0] and o[1] < p[1] for o in pts if o != p)]

def tam_igual(k, txt):
    pts = frente(txt)
    for (s1, k1), (s2, k2) in zip(pts, pts[1:]):
        if k >= k2 or (s2, k2) == pts[-1]:
            if k > pts[0][1]:
                (s1, k1), (s2, k2) = pts[0], pts[1]
            return s1 + (s2 - s1) * math.log(k / k1) / math.log(k2 / k1)

def bits(nome, k, txt):
    return (RAM_N[k] - tam_igual(kld[(f'{nome}_N{k}', txt)][0], txt)) * 2**30 * 8 / n_cam

T = ['# STRATA, C1: bits dos níveis escolhidos com a Hessiana (Llama-3.1-8B, caminho F16)', '',
     f'GPU: {GPU}. KL contra o BF16 em {KLD_CHUNKS} trechos de 512 tokens de cada texto (wikitext-2 raw de teste; C4 en, validação). '
     f'imatrix e Hessianas com {IMATRIX_CHUNKS} trechos do wikitext de treino. Matrizes das camadas: {n_cam / 1e9:.2f} bilhões de pesos. '
     f'Nível 0 = Q4_K_M com imatrix ({tam["Q4_K_M_imat"]:.3f} GiB); RAM do nível 1 {RAM_N[1]:.3f} GiB, do nível 2 {RAM_N[2]:.3f} GiB.', '',
     '## KL (×1000)', '', '| modelo | GiB | wikitext | C4 | mesmo 1º token (wiki) |', '|---|---:|---:|---:|---:|']
for nome in dict.fromkeys(n for n, _ in kld):
    s = tam.get(nome, RAM_N.get(int(nome[-1]) if nome[-2:-1] == 'N' else 0, float('nan')))
    w, c = kld.get((nome, 'wiki')), kld.get((nome, 'c4'))
    T.append(f'| {nome} | {s:.3f} | ' + (f'{1000 * w[0]:.2f} ± {1000 * w[1]:.2f}' if w else '') + ' | '
             + (f'{1000 * c[0]:.2f} ± {1000 * c[1]:.2f}' if c else '') + ' | ' + (f'{w[2]:.2f}%' if w else '') + ' |')
T += ['', '## Régua (bits/peso a mais que a frente dos k-quants em F16, no mesmo tamanho)', '',
      '| candidato | nível | wikitext | C4 | ganho sobre o P de hoje mantido no C4 |', '|---|---:|---:|---:|---:|']
veredito = {}
for pt, es in casos:
    nome = f'{pt}_{es}'
    for k in (1, 2):
        if (f'{nome}_N{k}', 'wiki') not in kld or (f'{nome}_N{k}', 'c4') not in kld:
            continue
        bw, bc = bits(nome, k, 'wiki'), bits(nome, k, 'c4')
        g = lambda t: math.log(kld[(f'P_perto_N{k}', t)][0] / kld[(f'{nome}_N{k}', t)][0])
        ret = g('c4') / g('wiki') if nome != 'P_perto' and g('wiki') > 0 else float('nan')
        veredito[(nome, k)] = (bw, bc, ret)
        T.append(f'| {nome} | {k} | {bw:+.3f} | {bc:+.3f} | ' + ('' if math.isnan(ret) else f'{100 * ret:.0f}%') + ' |')

def passa(nome):
    v = [veredito.get((nome, k)) for k in (1, 2)]
    return None if None in v else all(x[0] <= 0.05 for x in v) and all(x[2] >= 0.5 for x in v)
if passa('P_gptq'):
    V = 'SUCESSO: P com GPTQ ≤ +0,05 bit nos níveis 1 e 2 e o ganho se mantém no C4.'
elif all(veredito.get((n, 1), (1, 1, 0))[0] > 0.12 for n in ('P_gptq', 'D_gptq')):
    V = 'FRACASSO: nível 1 acima de +0,12 bit com P e com D.'
else:
    V = 'ZONA CINZA: ver a tabela (algum nível entre +0,05 e +0,12, só os D chegam lá, ou o ganho não se mantém no C4).'
T += ['', f'**Veredito pela régua:** {V}', '',
      '## Erro de saída das camadas (Σ ΔW H ΔWᵀ na calibração, relativo ao do nível 0)', '', '| candidato | nível 1 | nível 2 |', '|---|---:|---:|']
for n, (e1, e2) in erro.items():
    T.append(f'| {n} | {e1:.4f} | {e2:.4f} |')
T += ['', f'Tempo total: {(time.time() - T0) / 60:.0f} min.']
open('/content/res/strata_8b_gptq.md', 'w').write('\\n'.join(T) + '\\n')
json.dump({'gpu': GPU, 'kld': {f'{n}|{t}': v for (n, t), v in kld.items()}, 'tam': tam, 'ram_niveis': RAM_N, 'n_cam': n_cam,
           'erro_saida': erro, 'veredito': V}, open('/content/res/strata_8b_gptq.json', 'w'), indent=1)
display(Markdown('\\n'.join(T)))
"""),
    md("Cole a tabela na thread, ou baixe os arquivos abaixo (os `erro_*.json` têm o erro de saída por tensor)."),
    code("""
!cd /content/res && zip -q strata_8b_gptq.zip *
from google.colab import files
files.download('/content/res/strata_8b_gptq.zip')
"""),
]

nb = {"cells": celulas, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(AQUI, "strata_8b_gptq_colab.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print(out)
