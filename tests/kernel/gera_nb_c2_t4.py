#!/usr/bin/env python3
"""Gera tests/kernel/strata_8b_c2_t4_colab.ipynb: o C2 de docs/10 no Llama-3.1-8B numa T4 (15 GB), tudo no PyTorch.

    python tests/kernel/gera_nb_c2_t4.py SHA_DO_STRATA

O 8B em F16 (16 GB) não cabe na T4. O modelo roda camada a camada (`--fluxo`): as camadas ficam na RAM e só uma por vez
vai à GPU, com todos os trechos juntos. A KL de teste e a frente dos k-quants saem do mesmo caminho
(`strata_dkl.py teste`), contra o modelo original em fp16; o llama.cpp só quantiza e faz a imatrix (a partir do Q8_0,
que cabe na GPU).
"""
import json, os, sys

SHA = sys.argv[1]
AQUI = os.path.dirname(os.path.abspath(__file__))
RES = 'strata_8b_c2_t4'


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


celulas = [
    md("""
# STRATA, C2 numa T4: nível de cada grupo de matrizes escolhido pelo efeito medido na saída (Llama-3.1-8B)

**Antes de rodar:** *Ambiente de execução → Alterar o tipo de ambiente de execução* → **T4** com **RAM alta** (precisa de ~50 GB de RAM). Depois *Executar tudo*. Leva umas 4 a 5 horas e precisa de ~110 GB de disco.

O plano está em `docs/10-motor-de-compressao.md`. O nível 0 é o Q4_K_M com imatrix, intacto byte a byte. O nível 1 tem pontos P, e o nível 2 a tabela linha (DL), os dois com os bits do GPTQ (Hessianas de 1000 trechos, amortecimento 0,1).

**O que muda é o alocador.** Hoje, o `strata_mistura.py` escolhe o nível de cada matriz pelo erro com imatrix. Aqui, o `strata_dkl.py` mede quanto a KL da saída muda quando cada grupo (7 tipos de matriz × faixas de 4 camadas = 56 grupos) desce ou sobe um nível. Os textos dessa medida não se cruzam com os de teste nem com a calibração. Uma programação dinâmica escolhe o nível de cada grupo no orçamento, supondo que os efeitos somam.

**Por que este notebook é diferente do da L4.** O 8B em F16 (16 GB) não cabe nos 15 GB da T4. Então:
- o modelo roda camada a camada: as camadas ficam na RAM e só uma por vez vai à GPU, com todos os trechos juntos;
- a KL de teste também sai do PyTorch (`strata_dkl.py teste`), contra o modelo original em fp16, nos mesmos trechos que o llama-perplexity usaria (segunda metade de cada trecho de 512 tokens). A frente dos k-quants é medida aqui, no mesmo caminho;
- a imatrix é feita a partir do Q8_0, que cabe na GPU;
- as Hessianas saem de uma passada só, camada a camada, em fp16.

No 1B, a medida camada a camada deu a mesma KL do modelo inteiro e Hessianas iguais bit a bit. A KL do PyTorch foi conferida contra a do llama.cpp na frente, nos níveis e nas misturas (`tests/kernel/resultados/unsloth__Llama-3.2-1B_c2_container.md`).

**No Llama-3.2-1B** (KL ×1000 no wikitext / C4, llama.cpp, 40 trechos), em 850, 900 e 950 MiB:
- medido: 12,69 / 13,71, 7,28 / 8,13 e 4,78 / 5,79;
- de hoje: 22,91 / 24,23, 16,57 / 17,28 e 7,62 / 8,64;
- níveis inteiros interpolados: 13,99 / 15,40, 7,66 / 8,85 e 4,94 / 5,74.

**Régua (escrita antes de rodar, docs/10)**, em 5,33, 5,48 e 5,63 GiB, com KL em 200 trechos de cada texto de teste:
- **sucesso:** bits a mais da mistura ≤ atraso dos níveis inteiros interpolado em GiB + 0,10, nos dois textos; nenhum orçamento pior que o de hoje além de 1 barra de erro; e a soma prevista com erro ≤ 10%;
- **zona cinza:** ganha do de hoje, mas fica acima de +0,10;
- **fracasso:** não ganha do de hoje em pelo menos dois orçamentos.

O relatório também mostra os bits a mais contra a frente. No 1B, o alocador de hoje ficou pior até que os níveis inteiros, então o que interessa é fechar o atraso do nível 1 no C4 (+0,12 a +0,14 bit no 8B).
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
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
assert gpu.returncode == 0, 'Sem GPU NVIDIA: troque o tipo de ambiente de execução para T4.'
GPU = gpu.stdout.strip().splitlines()[0]
N = os.cpu_count()
RAM = int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1]) / 1e6
DISCO = shutil.disk_usage('/content').free / 1e9
print(f'GPU: {GPU} | CPU: {N} threads | RAM {RAM:.0f} GB | disco livre {DISCO:.0f} GB')
assert RAM > 45, 'Pouca RAM: ligue a RAM alta (o modelo fica inteiro na RAM, em fp16).'
if DISCO < 110:
    print('Aviso: pouco disco; pode faltar no fim.')
T0 = time.time()
"""),
    code(f"""
STRATA_SHA = "{SHA}"   # o commit do STRATA com o C2 camada a camada (o branch continua mudando)
LLAMA_SHA = "2a667032dc788c3a071d538e0b572e45a3f12731"
!rm -rf strata && git init -q strata && cd strata && git remote add origin https://github.com/Cantyll/strata.git && git fetch -q --depth 1 origin $STRATA_SHA && git checkout -q FETCH_HEAD
!rm -rf llama.cpp && git init -q llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp.git && git fetch -q --depth 1 origin $LLAMA_SHA && git checkout -q FETCH_HEAD
!pip install -q -U transformers safetensors
!pip install -q -e llama.cpp/gguf-py
!cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DGGML_NATIVE=ON 2>&1 | tail -3 && cmake --build build -j $(nproc) --target llama-quantize llama-imatrix 2>&1 | tail -5
!ls -la llama.cpp/build/bin | grep -E 'llama-(quantize|imatrix)$'
"""),
    code("""
import urllib.request, zipfile, io, gzip
from huggingface_hub import hf_hub_download
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip').read()))
for f in ['test', 'train', 'valid']:
    open(f'/content/wiki.{f}.raw', 'wb').write(z.read(f'wikitext-2-raw/wiki.{f}.raw'))
def c4(arq, ate):
    p = hf_hub_download('allenai/c4', f'en/c4-validation.{arq}-of-00008.json.gz', repo_type='dataset')
    txt, n = [], 0
    with gzip.open(p, 'rt') as f:
        for l in f:
            t = json.loads(l)['text']; txt.append(t); n += len(t)
            if n > ate:
                break
    return '\\n\\n'.join(txt)
# texto de teste: o começo do primeiro arquivo de validação do C4 (inglês), ~1,5 MB
open('/content/c4.test.raw', 'w').write(c4('00000', 1_500_000))
# calibração misturada (a mesma do C1): 1,3 milhão de caracteres do wikitext de treino e 1,3 milhão do C4 validação 00001,
# em blocos alternados de 20 KB. A medida do ΔKL usa o mesmo 00001 depois dos 2 milhões de caracteres.
c4c, wk = c4('00001', 3_000_000), open('/content/wiki.train.raw').read()[:1_300_000]
open('/content/calib_mix.raw', 'w').write(''.join(wk[i:i + 20_000] + c4c[i:i + 20_000] for i in range(0, 1_300_000, 20_000)))
open('/content/c4.medida.raw', 'w').write(c4c[2_000_000:3_000_000])
for f in ['wiki.test.raw', 'wiki.valid.raw', 'c4.test.raw', 'c4.medida.raw', 'calib_mix.raw']:
    print(f, os.path.getsize('/content/' + f), 'bytes')
"""),
    md("Modelo em BF16 (para os scripts do STRATA), Q8_0, imatrix a partir do Q8_0 na GPU, e o Q4_K_M com imatrix (o nível 0)."),
    code("""
from huggingface_hub import snapshot_download
src = snapshot_download(MODELO_HF, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
G = '/content/gguf'
os.makedirs(G, exist_ok=True)
B = 'llama.cpp/build/bin'
!python llama.cpp/convert_hf_to_gguf.py {src} --outtype bf16 --outfile {G}/BF16.gguf 2>&1 | tail -1
!{B}/llama-quantize {G}/BF16.gguf {G}/Q8_0.gguf Q8_0 > {G}/Q8_0.log 2>&1
!{B}/llama-imatrix -m {G}/Q8_0.gguf -f /content/calib_mix.raw -ngl 99 -c 512 --chunks {IMATRIX_CHUNKS} -o {G}/imatrix.gguf > {G}/imatrix.log 2>&1
!tail -2 {G}/imatrix.log
!{B}/llama-quantize --imatrix {G}/imatrix.gguf {G}/BF16.gguf {G}/Q4_K_M_imat.gguf Q4_K_M > {G}/Q4_K_M_imat.log 2>&1
!ls -la {G}/*.gguf
!df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Hessianas das entradas de cada matriz, numa passada só, camada a camada (≈ 1 GB por camada, em disco)."),
    code("""
!python strata/tests/kernel/strata_hessiana.py --modelo {src} --texto /content/calib_mix.raw --out /content/hess --chunks {HESS_CHUNKS} --device cuda --fluxo --dtype fp16 2>&1 | grep -vE 'Warning|Token indices|Loading' | tail -3
!du -sh /content/hess; df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Os três níveis em F16: o nível 0 desquantizado e os níveis 1 e 2 do GPTQ."),
    code("""
!python strata/tests/kernel/strata_escala_nivel.py --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf --gguf {G}/Q4_K_M_imat.gguf --modo copia --out /content/e0.gguf 2>&1 | grep -v Warning | tail -1
t = time.time()
out = !python strata/tests/kernel/strata_gptq_niveis.py --bf16 {G}/BF16.gguf --gguf {G}/Q4_K_M_imat.gguf --imatrix {G}/imatrix.gguf --hess /content/hess --amortece {AMORTECE} --device cuda --pontos P,DL --escolha gptq --out /content/e1.gguf /content/e2.gguf 2>&1
print(out[-1], f'({(time.time() - t) / 60:.0f} min)')
!rm -rf /content/hess
!df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("A KL de teste (`strata_dkl.py teste`): contra o modelo original, com a base de cada texto guardada uma vez."),
    code("""
kld, tam = {}, {}
TESTE = f'python strata/tests/kernel/strata_dkl.py teste --modelo {src} --texto /content/wiki.test.raw /content/c4.test.raw --chunks {KLD_CHUNKS} --device cuda --fluxo --base /content/base'
os.makedirs('/content/res', exist_ok=True)
def kl_de(nome, niveis, plano=''):
    t = time.time()
    out = !{TESTE} --niveis {niveis} {plano} --out /content/res/kl_{nome}.json 2>&1
    if not os.path.exists(f'/content/res/kl_{nome}.json'):
        print(f'Erro na KL de {nome}:', '\\n'.join(out[-8:]))
        return
    r = json.load(open(f'/content/res/kl_{nome}.json'))
    for txt, arq in (('wiki', 'wiki.test.raw'), ('c4', 'c4.test.raw')):
        kld[(nome, txt)] = (r[arq]['kl'], r[arq]['erro'], r[arq]['top1'])
    print(f'{nome}: wiki KL ×1000 {1000 * kld[(nome, "wiki")][0]:.2f}, c4 {1000 * kld[(nome, "c4")][0]:.2f} ({(time.time() - t) / 60:.0f} min)')
"""),
    md("A frente dos k-quants com a mesma imatrix, no mesmo caminho, e os níveis inteiros."),
    code("""
ESC = f'python strata/tests/kernel/strata_escala_nivel.py --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf'
tam['Q4_K_M_imat'] = os.path.getsize(f'{G}/Q4_K_M_imat.gguf') / 2**30
kl_de('Q4_K_M_imat', '/content/e0.gguf')
for q in ['Q5_K_S', 'Q5_K_M', 'Q6_K', 'Q8_0']:
    nome = q if q == 'Q8_0' else q + '_imat'
    if q != 'Q8_0':
        !{B}/llama-quantize --imatrix {G}/imatrix.gguf {G}/BF16.gguf {G}/{nome}.gguf {q} > {G}/{nome}.log 2>&1
    tam[nome] = os.path.getsize(f'{G}/{nome}.gguf') / 2**30
    !{ESC} --gguf {G}/{nome}.gguf --modo copia --out /content/q.gguf 2>&1 | grep -v Warning | tail -1
    kl_de(nome, '/content/q.gguf')
    !rm -f /content/q.gguf {G}/{nome}.gguf
for k in (1, 2):
    kl_de(f'nível {k}'.replace(' ', '_'), f'/content/e{k}.gguf')
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("ΔKL de cada grupo (textos da medida) e o erro com imatrix de cada matriz (o alocador de hoje)."),
    code("""
NIV = '--niveis /content/e0.gguf /content/e1.gguf /content/e2.gguf'
TXT = f'--modelo {src} {NIV} --texto /content/wiki.valid.raw /content/c4.medida.raw --chunks {DKL_CHUNKS} --device cuda --fluxo'
C = '/content/c2'
os.makedirs(C, exist_ok=True)
t = time.time()
!python strata/tests/kernel/strata_dkl.py mede {TXT} --banda {BANDA} --ref 1 --out {C}/dkl.json 2>&1 | grep -vE 'Warning|Token indices|copy_|Loading' | tail -60
print(f'mede: {(time.time() - t) / 60:.0f} min')
t = time.time()
!python strata/tests/demanda/strata_mistura.py erros --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf {NIV} --out {C}/erros.json 2>&1 | tail -1
print(f'erros: {(time.time() - t) / 60:.0f} min; {(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Os planos nos três orçamentos, medido e de hoje, e a KL de teste de cada um (a mistura monta em memória; nenhum GGUF é gravado)."),
    code("""
gib = {}
for o in ORCAMENTOS:
    !python strata/tests/kernel/strata_dkl.py plano --dkl {C}/dkl.json --q4km {G}/Q4_K_M_imat.gguf --gib {o} --tabela {TABELA} --out {C}/p_dkl_{o}.json
    !python strata/tests/demanda/strata_mistura.py grava --erros {C}/erros.json --q4km {G}/Q4_K_M_imat.gguf --gib {o} --tabela {TABELA} {NIV} --plano-out {C}/p_mse_{o}.json
    for a in ('dkl', 'mse'):
        gib[f'{a}_{o}'] = json.load(open(f'{C}/p_{a}_{o}.json'))['gib']
        kl_de(f'{a}_{o}', '/content/e0.gguf /content/e1.gguf /content/e2.gguf', f'--plano {C}/p_{a}_{o}.json')
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("A soma prevista contra a KL medida nos textos da medida."),
    code("""
PL = ' '.join(f'{C}/p_{a}_{o}.json' for a in ('dkl', 'mse') for o in ORCAMENTOS)
!python strata/tests/kernel/strata_dkl.py avalia {TXT} --plano {PL} --out {C}/avalia.json 2>&1 | grep 'KL medida'
avalia = json.load(open(f'{C}/avalia.json'))
erro_soma = {o: json.load(open(f'{C}/p_dkl_{o}.json'))['kl_prevista'] / avalia[f'{C}/p_dkl_{o}.json'] - 1 for o in ORCAMENTOS}
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("""
**A régua.** RAM do nível k = Q4_K_M + k bits por peso nas matrizes das camadas; a de uma mistura = Q4_K_M + os bytes que ela soma. *bits a mais* = quanto gasta além do k-quant de mesma KL (interpolação do log da KL entre os k-quants vizinhos). Abaixo de 0 = o STRATA ganha. *Níveis inteiros* = log da KL interpolado entre os níveis inteiros vizinhos, no mesmo tamanho.
"""),
    code("""
import math, gguf
from IPython.display import Markdown, display
r = gguf.GGUFReader(f'{G}/Q4_K_M_imat.gguf')
n_cam = sum(int(t.n_elements) for t in r.tensors if t.name.startswith('blk.') and t.tensor_type.name in ('Q4_K', 'Q6_K') and t.n_elements > 1e6)
KQ = ['Q4_K_M_imat', 'Q5_K_S_imat', 'Q5_K_M_imat', 'Q6_K_imat', 'Q8_0']
Q4 = tam['Q4_K_M_imat']
RAM_N = {0: Q4, 1: Q4 + n_cam / 8 / 2**30, 2: Q4 + 2 * n_cam / 8 / 2**30}
NOME_N = {0: 'Q4_K_M_imat', 1: 'nível_1', 2: 'nível_2'}

def frente(txt):
    pts = sorted((tam[q], kld[(q, txt)][0]) for q in KQ if (q, txt) in kld)
    return [p for p in pts if not any(o[0] <= p[0] and o[1] < p[1] for o in pts if o != p)]

def tam_igual(k, txt):
    pts = frente(txt)
    for (s1, k1), (s2, k2) in zip(pts, pts[1:]):
        if k >= k2 or (s2, k2) == pts[-1]:
            if k > pts[0][1]:
                (s1, k1), (s2, k2) = pts[0], pts[1]
            return s1 + (s2 - s1) * math.log(k / k1) / math.log(k2 / k1)

def bits(ram, k, txt):
    return (ram - tam_igual(k, txt)) * 2**30 * 8 / n_cam

def inteiros(ram, txt):
    k = 0 if ram < RAM_N[1] else 1
    a = (ram - RAM_N[k]) / (RAM_N[k + 1] - RAM_N[k])
    l0, l1 = (math.log(kld[(NOME_N[j], txt)][0]) for j in (k, k + 1))
    return math.exp(l0 + a * (l1 - l0))

ram = {n: Q4 + g - os.path.getsize(f'{G}/Q4_K_M_imat.gguf') / 2**30 for n, g in gib.items()}
T = ['# STRATA, C2: alocador pelo efeito medido na saída (Llama-3.1-8B, T4, KL no PyTorch)', '',
     f'GPU: {GPU}. KL contra o modelo original em fp16 (PyTorch, camada a camada), na segunda metade de {KLD_CHUNKS} trechos de 512 tokens de cada texto (wikitext-2 raw de teste; C4 en, validação 00000). '
     f'Frente dos k-quants medida no mesmo caminho; imatrix com {IMATRIX_CHUNKS} trechos a partir do Q8_0. Níveis: 1 = P, 2 = D linha, bits do GPTQ (Hessianas com {HESS_CHUNKS} trechos em fp16, amortecimento {AMORTECE}). '
     f'ΔKL medido em {DKL_CHUNKS} trechos de cada texto da medida, grupos de {BANDA} camadas. Matrizes das camadas: {n_cam / 1e9:.2f} bilhões de pesos. '
     f'Nível 0 = Q4_K_M com imatrix ({Q4:.3f} GiB); RAM do nível 1 {RAM_N[1]:.3f} GiB, do nível 2 {RAM_N[2]:.3f} GiB.', '',
     '## Frente e níveis inteiros', '', '| modelo | GiB | wikitext | C4 | mesmo 1º token (wiki) | bits a mais (wiki / C4) |', '|---|---:|---:|---:|---:|---:|']
for nome in KQ + ['nível_1', 'nível_2']:
    w, c = kld.get((nome, 'wiki')), kld.get((nome, 'c4'))
    if not (w and c):
        continue
    s = tam.get(nome, RAM_N.get(int(nome[-1]) if nome.startswith('nível') else 0))
    b = f'{bits(s, w[0], "wiki"):+.3f} / {bits(s, c[0], "c4"):+.3f}' if nome.startswith('nível') else ''
    T.append(f'| {nome} | {s:.3f} | {1000 * w[0]:.2f} ± {1000 * w[1]:.2f} | {1000 * c[0]:.2f} ± {1000 * c[1]:.2f} | {w[2]:.2f}% | {b} |')
T += ['', '## Misturas (KL ×1000; bits a mais contra a frente)', '',
      '| orçamento | alocador | GiB | wikitext | bits | C4 | bits | níveis inteiros (wiki / C4) | bits dos inteiros |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
res = {}
for o in ORCAMENTOS:
    for a in ('dkl', 'mse'):
        n = f'{a}_{o}'
        w, c = kld.get((n, 'wiki')), kld.get((n, 'c4'))
        if not (w and c):
            continue
        iw, ic = inteiros(ram[n], 'wiki'), inteiros(ram[n], 'c4')
        res[n] = dict(ram=ram[n], kl=(w[0], c[0]), err=(w[1], c[1]), bits=(bits(ram[n], w[0], 'wiki'), bits(ram[n], c[0], 'c4')),
                      inteiros=(iw, ic), bits_inteiros=(bits(ram[n], iw, 'wiki'), bits(ram[n], ic, 'c4')))
        v = res[n]
        T.append(f'| {o} | {"medido" if a == "dkl" else "de hoje"} | {ram[n]:.3f} | {1000 * w[0]:.2f} ± {1000 * w[1]:.2f} | {v["bits"][0]:+.3f} | '
                 f'{1000 * c[0]:.2f} ± {1000 * c[1]:.2f} | {v["bits"][1]:+.3f} | {1000 * iw:.2f} / {1000 * ic:.2f} | {v["bits_inteiros"][0]:+.3f} / {v["bits_inteiros"][1]:+.3f} |')
T += ['', '## A soma prevê a medida? (textos da medida)', '', '| orçamento | prevista | medida | erro |', '|---|---:|---:|---:|']
for o in ORCAMENTOS:
    p = json.load(open(f'{C}/p_dkl_{o}.json'))['kl_prevista']
    T.append(f'| {o} | {1000 * p:.3f} | {1000 * avalia[f"{C}/p_dkl_{o}.json"]:.3f} | {100 * erro_soma[o]:+.1f}% |')

# a régua do docs/10
ganha = [all(res[f'dkl_{o}']['kl'][i] < res[f'mse_{o}']['kl'][i] for i in (0, 1)) for o in ORCAMENTOS]
nao_pior = all(res[f'dkl_{o}']['kl'][i] <= res[f'mse_{o}']['kl'][i] + math.hypot(res[f'dkl_{o}']['err'][i], res[f'mse_{o}']['err'][i])
               for o in ORCAMENTOS for i in (0, 1))
dentro = all(res[f'dkl_{o}']['bits'][i] <= res[f'dkl_{o}']['bits_inteiros'][i] + 0.10 for o in ORCAMENTOS for i in (0, 1))
soma = all(abs(e) <= 0.10 for e in erro_soma.values())
if sum(ganha) < 2:
    V = 'FRACASSO: o medido não ganha do alocador de hoje em pelo menos dois orçamentos.'
elif dentro and nao_pior and soma:
    V = 'SUCESSO: a mistura fica ≤ +0,10 bit além dos níveis inteiros nos dois textos, nenhum orçamento pior que o de hoje e a soma prevista erra ≤ 10%.'
else:
    V = ('ZONA CINZA: ganha do de hoje, mas ' + ', '.join(x for x, ok in (('passa de +0,10 bit além dos níveis inteiros', dentro),
         ('algum orçamento fica pior que o de hoje além de 1 barra', nao_pior), ('a soma prevista erra mais de 10%', soma)) if not ok) + '.')
T += ['', f'**Veredito pela régua:** {V}', '',
      'Ganho do medido sobre os níveis inteiros, em bits (wiki / C4): ' + ', '.join(
          f'{o} GiB {res[f"dkl_{o}"]["bits_inteiros"][0] - res[f"dkl_{o}"]["bits"][0]:+.3f} / {res[f"dkl_{o}"]["bits_inteiros"][1] - res[f"dkl_{o}"]["bits"][1]:+.3f}'
          for o in ORCAMENTOS) + '.', '', f'Tempo total: {(time.time() - T0) / 60:.0f} min.']
open('/content/res/{RES}.md', 'w').write('\\n'.join(T) + '\\n')
json.dump({'gpu': GPU, 'kld': {f'{n}|{t}': v for (n, t), v in kld.items()}, 'tam': tam, 'ram_niveis': RAM_N, 'n_cam': n_cam,
           'misturas': res, 'erro_soma': erro_soma, 'veredito': V}, open('/content/res/{RES}.json', 'w'), indent=1)
!cp {C}/*.json /content/res/
display(Markdown('\\n'.join(T)))
"""),
    md("Cole a tabela na thread, ou baixe os arquivos abaixo (`dkl.json` tem o ΔKL de cada grupo e os `p_*.json`, os planos)."),
    code("""
!cd /content/res && zip -q {RES}.zip *
from google.colab import files
files.download('/content/res/{RES}.zip')
"""),
]

for c in celulas:
    c["source"] = [l.replace("{RES}", RES) for l in c["source"]]

nb = {"cells": celulas, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "T4"},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(AQUI, RES + "_colab.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print(out)
