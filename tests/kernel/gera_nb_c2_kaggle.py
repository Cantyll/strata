#!/usr/bin/env python3
"""Gera tests/kernel/strata_8b_c2_kaggle.ipynb: o C2 de docs/10 no Llama-3.1-8B num notebook do Kaggle (T4, ~30 GB de
RAM, disco temporário de ~60–70 GB, sessão de até 12 h).

    python tests/kernel/gera_nb_c2_kaggle.py SHA_DO_STRATA

O mesmo caminho de medida do notebook da T4 do Colab (gera_nb_c2_t4.py: modelo camada a camada, KL de teste e frente no
PyTorch), em outra ordem, para caber no disco: a frente sai primeiro e cada k-quant é apagado depois de medido; o nível
0 é lido do próprio Q4_K_M (sem a cópia de 16 GB em F16); as Hessianas guardam só o triângulo de cima e o GPTQ apaga as
de cada camada depois de usá-las; os pesos do Hugging Face saem do disco durante o GPTQ e voltam depois.
"""
import json, os, sys

import gera_nb_c2_t4 as t4

SHA = sys.argv[1]
AQUI = os.path.dirname(os.path.abspath(__file__))
RES = 'strata_8b_c2_kaggle'
md, code = t4.md, t4.code

# a régua e os textos são os do notebook da T4, com os caminhos do Kaggle
REGUA = next(''.join(c['source']) for c in t4.celulas if c['cell_type'] == 'code' and 'def frente' in ''.join(c['source']))
REGUA = (REGUA.replace("f'{G}/Q4_K_M_imat.gguf'", "Q4KM").replace('/content/res', '/kaggle/working/res')
         .replace('(Llama-3.1-8B, T4, KL no PyTorch)', '(Llama-3.1-8B, Kaggle, KL no PyTorch)'))
assert 'Q4KM' in REGUA and '/content' not in REGUA
TEXTOS = next(''.join(c['source']) for c in t4.celulas if c['cell_type'] == 'code' and 'calib_mix' in ''.join(c['source'])
              and 'wikitext-2-raw-v1.zip' in ''.join(c['source'])).replace("'/content/", "W + '/").replace("f'/content/", "f'{W}/")
assert '/content' not in TEXTOS

celulas = [
    md("""
# STRATA, C2 no Kaggle: nível de cada grupo de matrizes escolhido pelo efeito medido na saída (Llama-3.1-8B)

**Antes de rodar**, no painel da direita (*Settings*):
- **Accelerator → GPU T4 x2** (usa uma das duas);
- **Internet → On** (o Kaggle pede o celular verificado na conta);
- **Persistence → No persistence**.

Depois, **Save Version → Save & Run All (Commit)**. Assim o notebook roda sozinho, mesmo com a aba fechada, até 12 h. Leva umas 6 a 8 horas. No fim, os resultados ficam na aba *Output* da versão salva (`res/strata_8b_c2_kaggle.zip`).

O plano está em `docs/10-motor-de-compressao.md`. É o mesmo experimento e o mesmo caminho de medida do notebook da T4 do Colab:
- o nível 0 é o Q4_K_M com imatrix, intacto byte a byte;
- o nível 1 tem pontos P e o nível 2 a tabela linha (DL), os dois com os bits do GPTQ (Hessianas de 1000 trechos, amortecimento 0,1);
- o `strata_dkl.py` mede quanto a KL muda quando cada grupo (7 tipos de matriz × faixas de 4 camadas) desce ou sobe um nível;
- uma programação dinâmica escolhe os níveis no orçamento;
- o modelo roda camada a camada (o 8B em F16 não cabe numa T4), e a KL de teste e a frente dos k-quants saem do PyTorch, contra o modelo original. No 1B, esse caminho deu os mesmos bits a mais que o llama.cpp, dentro de 0,013 bit.

**O que muda no Kaggle é só a ordem, para caber em ~60 GB de disco e ~30 GB de RAM.**
- A frente sai primeiro, e cada k-quant é apagado depois de medido.
- O nível 0 é lido do próprio Q4_K_M.
- As Hessianas guardam só o triângulo de cima, e o GPTQ apaga as de cada camada depois de usá-las.
- Os pesos do Hugging Face saem do disco durante o GPTQ e são baixados de novo depois.

**Régua (escrita antes de rodar, docs/10)**, em 5,33, 5,48 e 5,63 GiB, com KL em 200 trechos de cada texto de teste:
- **sucesso:** bits a mais da mistura ≤ atraso dos níveis inteiros interpolado em GiB + 0,10, nos dois textos; nenhum orçamento pior que o de hoje além de 1 barra de erro; e a soma prevista com erro ≤ 10%;
- **zona cinza:** ganha do de hoje, mas fica acima de +0,10;
- **fracasso:** não ganha do de hoje em pelo menos dois orçamentos.
"""),
    code("""
KLD_CHUNKS = 200           # trechos de 512 tokens na KL de teste, em cada texto
IMATRIX_CHUNKS = 500       # trechos da calibração misturada, para a imatrix
HESS_CHUNKS = 1000         # trechos para as Hessianas
AMORTECE = 0.1             # amortecimento do GPTQ
DKL_CHUNKS = 8             # trechos de cada texto da medida do ΔKL por grupo
BANDA = 4                  # camadas por grupo
ORCAMENTOS = [5.33, 5.48, 5.63]   # GiB
TABELA = 16                # bytes por matriz e nível da tabela do DL
MODELO_HF = "unsloth/Meta-Llama-3.1-8B"
"""),
    code("""
import os, re, json, shutil, subprocess, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'          # uma T4 só
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
assert gpu.returncode == 0, 'Sem GPU: em Settings, Accelerator → GPU T4 x2.'
GPU = gpu.stdout.strip().splitlines()[0]
N = os.cpu_count()
RAM = int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1]) / 1e6
# a pasta de trabalho: a de mais disco livre fora de /kaggle/working (que guarda só a saída, ~20 GB)
cand = [d for d in ('/kaggle/tmp', '/tmp', '/kaggle/working') if os.path.isdir(d) or d == '/kaggle/tmp']
os.makedirs('/kaggle/tmp', exist_ok=True)
livre = {d: shutil.disk_usage(d).free / 1e9 for d in cand if os.path.isdir(d)}
W = max((d for d in livre if d != '/kaggle/working'), key=livre.get)
os.environ['HF_HOME'] = W + '/hf_cache'
print(f'GPU: {GPU} | CPU: {N} threads | RAM {RAM:.0f} GB | disco livre ' + ', '.join(f'{d} {v:.0f} GB' for d, v in livre.items()) + f' | trabalho em {W}')
assert RAM > 25, 'Pouca RAM: o modelo fica inteiro na RAM, em fp16 (16 GB).'
if livre[W] < 60:
    print(f'AVISO: {livre[W]:.0f} GB livres; o pico de disco é ~57 GB. Pode faltar.')
!nvcc --version | tail -1
T0 = time.time()
"""),
    code(f"""
STRATA_SHA = "{SHA}"   # o commit do STRATA com o C2 (o branch continua mudando)
LLAMA_SHA = "2a667032dc788c3a071d538e0b572e45a3f12731"
%cd {{W}}
!rm -rf strata && git init -q strata && cd strata && git remote add origin https://github.com/Cantyll/strata.git && git fetch -q --depth 1 origin $STRATA_SHA && git checkout -q FETCH_HEAD
!rm -rf llama.cpp && git init -q llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch -q --depth 1 origin $LLAMA_SHA && git checkout -q FETCH_HEAD
!pip install -q -U transformers safetensors
!pip install -q -e llama.cpp/gguf-py
!cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_NATIVE=ON 2>&1 | tail -3 && cmake --build build -j $(nproc) --target llama-quantize llama-imatrix 2>&1 | tail -5
!ls -la llama.cpp/build/bin | grep -E 'llama-(quantize|imatrix)$'
print(f'{{(time.time() - T0) / 60:.0f}} min até aqui')
"""),
    code(TEXTOS),
    md("Modelo em BF16 (para os scripts do STRATA), Q8_0, imatrix a partir do Q8_0 na GPU e o Q4_K_M com imatrix (o nível 0)."),
    code("""
from huggingface_hub import snapshot_download
HF = W + '/hf8b'
baixa = lambda: snapshot_download(MODELO_HF, allow_patterns=["*.json", "*.safetensors", "tokenizer*"], local_dir=HF)
src = baixa()
G = W + '/gguf'
os.makedirs(G, exist_ok=True)
B = 'llama.cpp/build/bin'
Q4KM = f'{G}/Q4_K_M_imat.gguf'
!python llama.cpp/convert_hf_to_gguf.py {src} --outtype bf16 --outfile {G}/BF16.gguf 2>&1 | tail -1
!{B}/llama-quantize {G}/BF16.gguf {G}/Q8_0.gguf Q8_0 > {G}/Q8_0.log 2>&1
!{B}/llama-imatrix -m {G}/Q8_0.gguf -f {W}/calib_mix.raw -ngl 99 -c 512 --chunks {IMATRIX_CHUNKS} -o {G}/imatrix.gguf > {G}/imatrix.log 2>&1
!tail -2 {G}/imatrix.log
!{B}/llama-quantize --imatrix {G}/imatrix.gguf {G}/BF16.gguf {Q4KM} Q4_K_M > {G}/Q4_K_M_imat.log 2>&1
!ls -la {G}/*.gguf
!df -h {W} | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("A KL de teste (`strata_dkl.py teste`, camada a camada): contra o modelo original, com a base de cada texto guardada uma vez."),
    code("""
kld, tam = {}, {}
RESD = '/kaggle/working/res'
os.makedirs(RESD, exist_ok=True)
TESTE = f'python strata/tests/kernel/strata_dkl.py teste --modelo {HF} --texto {W}/wiki.test.raw {W}/c4.test.raw --chunks {KLD_CHUNKS} --device cuda --fluxo --base {W}/base'
def kl_de(nome, niveis, plano=''):
    t = time.time()
    out = !{TESTE} --niveis {niveis} {plano} --out {RESD}/kl_{nome}.json 2>&1
    if not os.path.exists(f'{RESD}/kl_{nome}.json'):
        print(f'Erro na KL de {nome}:', '\\n'.join(out[-8:]))
        return
    r = json.load(open(f'{RESD}/kl_{nome}.json'))
    for txt, arq in (('wiki', 'wiki.test.raw'), ('c4', 'c4.test.raw')):
        kld[(nome, txt)] = (r[arq]['kl'], r[arq]['erro'], r[arq]['top1'])
    print(f'{nome}: wiki KL ×1000 {1000 * kld[(nome, "wiki")][0]:.2f}, c4 {1000 * kld[(nome, "c4")][0]:.2f} ({(time.time() - t) / 60:.0f} min)')
"""),
    md("A frente dos k-quants com a mesma imatrix, lidos direto dos GGUFs quantizados; cada um é apagado depois de medido."),
    code("""
for nome, q in [('Q4_K_M_imat', None), ('Q8_0', None), ('Q5_K_S_imat', 'Q5_K_S'), ('Q5_K_M_imat', 'Q5_K_M'), ('Q6_K_imat', 'Q6_K')]:
    if q:
        !{B}/llama-quantize --imatrix {G}/imatrix.gguf {G}/BF16.gguf {G}/{nome}.gguf {q} > {G}/{nome}.log 2>&1
    tam[nome] = os.path.getsize(f'{G}/{nome}.gguf') / 2**30
    kl_de(nome, f'{G}/{nome}.gguf')
    if nome != 'Q4_K_M_imat':
        !rm -f {G}/{nome}.gguf
!df -h {W} | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Hessianas numa passada só, camada a camada, só o triângulo de cima (~0,5 GB por camada). Depois os pesos do Hugging Face saem do disco, e o GPTQ apaga as Hessianas de cada camada que termina."),
    code("""
!python strata/tests/kernel/strata_hessiana.py --modelo {HF} --texto {W}/calib_mix.raw --out {W}/hess --chunks {HESS_CHUNKS} --device cuda --fluxo --dtype fp16 --triangulo 2>&1 | grep -vE 'Warning|Token indices|Loading' | tail -3
!du -sh {W}/hess; df -h {W} | tail -1
!rm -rf {HF}
t = time.time()
out = !python strata/tests/kernel/strata_gptq_niveis.py --bf16 {G}/BF16.gguf --gguf {Q4KM} --imatrix {G}/imatrix.gguf --hess {W}/hess --apaga-hess --amortece {AMORTECE} --device cuda --pontos P,DL --escolha gptq --out {W}/e1.gguf {W}/e2.gguf 2>&1
print(out[-1], f'({(time.time() - t) / 60:.0f} min)')
!rm -rf {W}/hess
NIV = f'--niveis {Q4KM} {W}/e1.gguf {W}/e2.gguf'
C = W + '/c2'
os.makedirs(C, exist_ok=True)
t = time.time()
!python strata/tests/demanda/strata_mistura.py erros --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf {NIV} --out {C}/erros.json 2>&1 | tail -1
print(f'erros: {(time.time() - t) / 60:.0f} min')
!rm -f {G}/BF16.gguf
src = baixa()
!df -h {W} | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Os níveis inteiros, o ΔKL de cada grupo (textos da medida) e os planos nos três orçamentos, medido e de hoje (a mistura monta em memória; nenhum GGUF é gravado)."),
    code("""
for k in (1, 2):
    kl_de(f'nível_{k}', f'{W}/e{k}.gguf')
TXT = f'--modelo {HF} {NIV} --texto {W}/wiki.valid.raw {W}/c4.medida.raw --chunks {DKL_CHUNKS} --device cuda --fluxo'
t = time.time()
!python strata/tests/kernel/strata_dkl.py mede {TXT} --banda {BANDA} --ref 1 --out {C}/dkl.json 2>&1 | grep -vE 'Warning|Token indices|copy_|Loading' | tail -60
print(f'mede: {(time.time() - t) / 60:.0f} min')
gib = {}
for o in ORCAMENTOS:
    !python strata/tests/kernel/strata_dkl.py plano --dkl {C}/dkl.json --q4km {Q4KM} --gib {o} --tabela {TABELA} --out {C}/p_dkl_{o}.json
    !python strata/tests/demanda/strata_mistura.py grava --erros {C}/erros.json --q4km {Q4KM} --gib {o} --tabela {TABELA} {NIV} --plano-out {C}/p_mse_{o}.json
    for a in ('dkl', 'mse'):
        gib[f'{a}_{o}'] = json.load(open(f'{C}/p_{a}_{o}.json'))['gib']
        kl_de(f'{a}_{o}', f'{Q4KM} {W}/e1.gguf {W}/e2.gguf', f'--plano {C}/p_{a}_{o}.json')
PL = ' '.join(f'{C}/p_{a}_{o}.json' for a in ('dkl', 'mse') for o in ORCAMENTOS)
!python strata/tests/kernel/strata_dkl.py avalia {TXT} --plano {PL} --out {C}/avalia.json 2>&1 | grep 'KL medida'
avalia = json.load(open(f'{C}/avalia.json'))
erro_soma = {o: json.load(open(f'{C}/p_dkl_{o}.json'))['kl_prevista'] / avalia[f'{C}/p_dkl_{o}.json'] - 1 for o in ORCAMENTOS}
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("""
**A régua.** RAM do nível k = Q4_K_M + k bits por peso nas matrizes das camadas; a de uma mistura = Q4_K_M + os bytes que ela soma. *bits a mais* = quanto gasta além do k-quant de mesma KL (interpolação do log da KL entre os k-quants vizinhos). Abaixo de 0 = o STRATA ganha. *Níveis inteiros* = log da KL interpolado entre os níveis inteiros vizinhos, no mesmo tamanho.
"""),
    code(REGUA),
    md("Os resultados ficam na aba *Output* da versão salva. Cole a tabela na thread, ou baixe o zip abaixo."),
    code("""
!cd /kaggle/working/res && zip -q {RES}.zip *.md *.json
print('/kaggle/working/res/{RES}.zip')
"""),
]

for c in celulas:
    c["source"] = [l.replace("{RES}", RES) for l in c["source"]]

nb = {"cells": celulas, "metadata": {"accelerator": "GPU", "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                                                    "isGpuEnabled": True, "language": "python"},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                                      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = os.path.join(AQUI, RES + ".ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print(out)
