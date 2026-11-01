#!/usr/bin/env python3
"""Gera tests/kernel/strata_8b_c2_colab.ipynb (C2 de docs/10 no Llama-3.1-8B: alocador pelo efeito medido na saída).

    python tests/kernel/gera_nb_c2.py SHA_DO_STRATA

A frente dos k-quants vem da rodada do C1b (unsloth__Meta-Llama-3.1-8B_gptq_c1b_a100.json); o Q4_K_M é medido de novo
para conferir a base.
"""
import json, os, sys

SHA = sys.argv[1]
AQUI = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(AQUI, 'resultados', 'unsloth__Meta-Llama-3.1-8B_gptq_c1b_a100.json')))
KQ = ['Q4_K_M_imat', 'Q5_K_S_imat', 'Q5_K_M_imat', 'Q6_K_imat', 'Q8_0']
REF = {'kld': {k: v for k, v in REF['kld'].items() if k.split('|')[0] in KQ}, 'tam': {k: REF['tam'][k] for k in KQ}}
RES = 'strata_8b_c2'


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


celulas = [
    md("""
# STRATA, C2: nível de cada grupo de matrizes escolhido pelo efeito medido na saída (Llama-3.1-8B)

**Antes de rodar:** *Ambiente de execução → Alterar o tipo de ambiente de execução* → **L4** (mais barata em unidades, ~6 h) ou **A100** (~3,5 h), com **RAM alta**. Depois *Executar tudo*. Precisa de ~150 GB de disco.

O plano está em `docs/10-motor-de-compressao.md`. O nível 0 é o Q4_K_M com imatrix, intacto byte a byte. Os níveis são os do fim do C1: o nível 1 com pontos P e o nível 2 com a tabela linha (DL), ambos com os bits do GPTQ (Hessianas de 1000 trechos, amortecimento 0,1). O formato e o kernel não mudam.

**O que muda é o alocador.** Hoje, quem escolhe o nível de cada matriz é o `strata_mistura.py`, pelo erro com imatrix. Aqui, o `strata_dkl.py` roda o modelo do Hugging Face com os pesos dos níveis trocados em memória e mede quanto a KL da saída muda quando cada grupo (7 tipos de matriz × faixas de 4 camadas = 56 grupos) desce ou sobe um nível. Uma programação dinâmica escolhe o nível de cada grupo no orçamento, supondo que os efeitos somam. Os textos dessa medida (wikitext-2 de validação e C4 validação 00001 depois dos 2 milhões de caracteres) não se cruzam com os de teste nem com a calibração.

**No Llama-3.2-1B** (contêiner, KL ×1000 no wikitext / C4, 40 trechos), em 850, 900 e 950 MiB:
- medido: 12,69 / 13,71, 7,28 / 8,13 e 4,78 / 5,79;
- de hoje: 22,91 / 24,23, 16,57 / 17,28 e 7,62 / 8,64;
- níveis inteiros interpolados: 13,99 / 15,40, 7,66 / 8,85 e 4,94 / 5,74.

A soma prevista errou +1,7%, +0,7% e −7,3%. O 1B já prometeu mais do que o 8B entregou pelo menos quatro vezes, e quem decide é este notebook.

**Régua (escrita antes de rodar, docs/10)**, em 5,33, 5,48 e 5,63 GiB, com KL em 200 trechos de cada texto de teste:
- **sucesso:** bits a mais da mistura ≤ atraso dos níveis inteiros interpolado em GiB + 0,10, nos dois textos; nenhum orçamento pior que o de hoje além de 1 barra de erro; e a soma prevista com erro ≤ 10%;
- **zona cinza:** ganha do de hoje, mas fica acima de +0,10;
- **fracasso:** não ganha do de hoje em pelo menos dois orçamentos.

O relatório também mostra os bits a mais contra a frente dos k-quants. No 1B, o alocador de hoje ficou pior até que os níveis inteiros, então "ganhar do de hoje" é uma barra baixa. O que interessa é fechar o atraso do nível 1 no C4 (+0,12 a +0,14 bit no 8B).
"""),
    code("""
KLD_CHUNKS = 200           # trechos na KL de teste, em cada texto (~65 MB de disco por trecho)
IMATRIX_CHUNKS = 500       # trechos de 512 tokens da calibração misturada, para a imatrix
HESS_CHUNKS = 1000         # trechos para as Hessianas
AMORTECE = 0.1             # amortecimento do GPTQ
GRUPO = 0                  # camadas por passada na coleta das Hessianas (0 = 8 numa GPU de 40 GB ou mais, 3 abaixo)
DKL_CHUNKS = 8             # trechos de cada texto da medida do ΔKL por grupo
BANDA = 4                  # camadas por grupo
ORCAMENTOS = [5.33, 5.48, 5.63]   # GiB
TABELA = 16                # bytes por matriz e nível da tabela do DL
REF = {REF}
MODELO_HF = "unsloth/Meta-Llama-3.1-8B"
""".replace('{REF}', repr(REF))),
    code("""
import os, re, json, shutil, subprocess, time
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', '--format=csv,noheader'], capture_output=True, text=True)
assert gpu.returncode == 0, 'Sem GPU NVIDIA: troque o tipo de ambiente de execução para L4 ou A100.'
GPU = gpu.stdout.strip().splitlines()[0]
GRUPO = GRUPO or (8 if int(re.search(r'(\\d+) MiB', GPU).group(1)) >= 40000 else 3)
N = os.cpu_count()
RAM = int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1]) / 1e6
DISCO = shutil.disk_usage('/content').free / 1e9
print(f'GPU: {GPU} | CPU: {N} threads | RAM {RAM:.0f} GB | disco livre {DISCO:.0f} GB')
if DISCO < 150:
    print('Aviso: pouco disco. Ponha KLD_CHUNKS = 100.')
if RAM < 50:
    print('Aviso: pouca RAM. Ligue a RAM alta.')
T0 = time.time()
"""),
    code(f"""
STRATA_SHA = "{SHA}"   # o commit do STRATA com o C2 (o branch continua mudando)
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
for f in ['test', 'train', 'valid']:
    open(f'/content/wiki.{f}.raw', 'wb').write(z.read(f'wikitext-2-raw/wiki.{f}.raw'))
# texto de teste: o começo do primeiro arquivo de validação do C4 (inglês), ~1,5 MB
def c4(arq, ate):
    p = hf_hub_download('allenai/c4', f'en/c4-validation.{arq}-of-00008.json.gz', repo_type='dataset')
    txt, n = [], 0
    with gzip.open(p, 'rt') as f:
        for l in f:
            t = json.loads(l)['text']; txt.append(t); n += len(t)
            if n > ate:
                break
    return '\\n\\n'.join(txt)
open('/content/c4.test.raw', 'w').write(c4('00000', 1_500_000))
# calibração misturada (a mesma do C1): 1,3 milhão de caracteres do wikitext de treino e 1,3 milhão do C4 validação 00001,
# em blocos alternados de 20 KB. A medida do ΔKL usa o mesmo 00001 depois dos 2 milhões de caracteres.
c4c, wk = c4('00001', 3_000_000), open('/content/wiki.train.raw').read()[:1_300_000]
open('/content/calib_mix.raw', 'w').write(''.join(wk[i:i + 20_000] + c4c[i:i + 20_000] for i in range(0, 1_300_000, 20_000)))
open('/content/c4.medida.raw', 'w').write(c4c[2_000_000:3_000_000])
for f in ['wiki.test.raw', 'wiki.valid.raw', 'c4.test.raw', 'c4.medida.raw', 'calib_mix.raw']:
    print(f, os.path.getsize('/content/' + f), 'bytes')
"""),
    md("Modelo em BF16, imatrix, Q4_K_M com imatrix e as duas bases da KL. Os pesos no formato do Hugging Face ficam até o fim: o `strata_dkl.py` roda neles."),
    code("""
from huggingface_hub import snapshot_download
src = snapshot_download(MODELO_HF, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
G = '/content/gguf'
os.makedirs(G, exist_ok=True)
B = 'llama.cpp/build/bin'
!python llama.cpp/convert_hf_to_gguf.py {src} --outtype bf16 --outfile {G}/BF16.gguf 2>&1 | tail -1
!{B}/llama-imatrix -m {G}/BF16.gguf -f /content/calib_mix.raw -ngl 99 -c 512 --chunks {IMATRIX_CHUNKS} -o {G}/imatrix.gguf > {G}/imatrix.log 2>&1
!tail -2 {G}/imatrix.log
!{B}/llama-quantize --imatrix {G}/imatrix.gguf {G}/BF16.gguf {G}/Q4_K_M_imat.gguf Q4_K_M > {G}/Q4_K_M_imat.log 2>&1
BASE = {'wiki': '/content/kld_wiki.bin', 'c4': '/content/kld_c4.bin'}
!{B}/llama-perplexity -m {G}/BF16.gguf -f /content/wiki.test.raw -c 512 -ngl 99 --chunks {KLD_CHUNKS} --kl-divergence-base {BASE['wiki']} > /dev/null 2>&1
!{B}/llama-perplexity -m {G}/BF16.gguf -f /content/c4.test.raw -c 512 -ngl 99 --chunks {KLD_CHUNKS} --kl-divergence-base {BASE['c4']} > /dev/null 2>&1
!ls -la {G}/*.gguf /content/kld_*.bin
!df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Hessianas das entradas de cada matriz, com os mesmos trechos da imatrix (≈ 1 GB por camada, em disco)."),
    code("""
!python strata/tests/kernel/strata_hessiana.py --modelo {src} --texto /content/calib_mix.raw --out /content/hess --chunks {HESS_CHUNKS} --device cuda --grupo {GRUPO} 2>&1 | grep -v Warning | tail -3
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
    md("Os três níveis em F16. O nível 0 desquantizado mede de novo o Q4_K_M, para conferir que a base é a mesma da frente reaproveitada."),
    code("""
import gguf
tam = dict(REF['tam'])
for k_, v_ in REF['kld'].items():
    kld[tuple(k_.split('|'))] = tuple(v_)
!python strata/tests/kernel/strata_escala_nivel.py --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf --gguf {G}/Q4_K_M_imat.gguf --modo copia --out /content/e0.gguf 2>&1 | grep -v Warning | tail -1
kl_de('/content/e0.gguf', 'Q4_K_M_imat_agora')
tam['Q4_K_M_imat_agora'] = os.path.getsize(f'{G}/Q4_K_M_imat.gguf') / 2**30
t = time.time()
out = !python strata/tests/kernel/strata_gptq_niveis.py --bf16 {G}/BF16.gguf --gguf {G}/Q4_K_M_imat.gguf --imatrix {G}/imatrix.gguf --hess /content/hess --amortece {AMORTECE} --device cuda --pontos P,DL --escolha gptq --out /content/e1.gguf /content/e2.gguf 2>&1
print(out[-1], f'({(time.time() - t) / 60:.0f} min)')
for k in (1, 2):
    kl_de(f'/content/e{k}.gguf', f'nível {k}')
!rm -rf /content/hess
!df -h /content | tail -1
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("ΔKL de cada grupo, medido no PyTorch (GPU, fp16), e o erro com imatrix de cada matriz (o alocador de hoje)."),
    code("""
NIV = '--niveis /content/e0.gguf /content/e1.gguf /content/e2.gguf'
TXT = f'--modelo {src} {NIV} --texto /content/wiki.valid.raw /content/c4.medida.raw --chunks {DKL_CHUNKS} --device cuda'
C = '/content/c2'
os.makedirs(C, exist_ok=True)
t = time.time()
!python strata/tests/kernel/strata_dkl.py mede {TXT} --banda {BANDA} --ref 1 --out {C}/dkl.json 2>&1 | grep -vE 'Warning|Token indices|copy_' | tail -60
print(f'mede: {(time.time() - t) / 60:.0f} min')
t = time.time()
!python strata/tests/demanda/strata_mistura.py erros --bf16 {G}/BF16.gguf --imatrix {G}/imatrix.gguf {NIV} --out {C}/erros.json 2>&1 | tail -1
print(f'erros: {(time.time() - t) / 60:.0f} min; {(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("Os planos nos três orçamentos: medido (`strata_dkl.py plano`) e o de hoje (`strata_mistura.py`). Cada GGUF misturado é gravado, medido e apagado."),
    code("""
gib = {}
for o in ORCAMENTOS:
    !python strata/tests/kernel/strata_dkl.py plano --dkl {C}/dkl.json --q4km {G}/Q4_K_M_imat.gguf --gib {o} --tabela {TABELA} --out {C}/p_dkl_{o}.json
    for a in ('dkl', 'mse'):
        op = f'--plano {C}/p_dkl_{o}.json' if a == 'dkl' else f'--plano-out {C}/p_mse_{o}.json'
        !python strata/tests/demanda/strata_mistura.py grava --erros {C}/erros.json --q4km {G}/Q4_K_M_imat.gguf --gib {o} --tabela {TABELA} {NIV} {op} --out /content/mix.gguf 2>&1 | tail -1
        gib[f'{a}_{o}'] = json.load(open(f'{C}/p_{a}_{o}.json'))['gib']
        kl_de('/content/mix.gguf', f'{a}_{o}')
        !rm -f /content/mix.gguf
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("A soma prevista contra a KL medida no PyTorch, nos mesmos textos da medida."),
    code("""
PL = ' '.join(f'{C}/p_{a}_{o}.json' for a in ('dkl', 'mse') for o in ORCAMENTOS)
!python strata/tests/kernel/strata_dkl.py avalia {TXT} --plano {PL} --out {C}/avalia.json 2>&1 | grep 'KL medida'
avalia = json.load(open(f'{C}/avalia.json'))
erro_soma = {o: json.load(open(f'{C}/p_dkl_{o}.json'))['kl_prevista'] / avalia[f'{C}/p_dkl_{o}.json'] - 1 for o in ORCAMENTOS}
print(f'{(time.time() - T0) / 60:.0f} min até aqui')
"""),
    md("""
**A régua.** RAM do nível k = Q4_K_M + k bits por peso nas matrizes das camadas; a de uma mistura = Q4_K_M + os bytes que ela soma. *bits a mais* = quanto gasta além do k-quant em F16 de mesma KL (interpolação do log da KL entre os k-quants vizinhos). Abaixo de 0 = o STRATA ganha. *Níveis inteiros* = log da KL interpolado entre os níveis inteiros vizinhos, no mesmo tamanho.
"""),
    code("""
import math
from IPython.display import Markdown, display
r = gguf.GGUFReader(f'{G}/Q4_K_M_imat.gguf')
n_cam = sum(int(t.n_elements) for t in r.tensors if t.name.startswith('blk.') and t.tensor_type.name in ('Q4_K', 'Q6_K') and t.n_elements > 1e6)
KQ = ['Q4_K_M_imat', 'Q5_K_S_imat', 'Q5_K_M_imat', 'Q6_K_imat', 'Q8_0']
Q4 = tam['Q4_K_M_imat_agora']
RAM_N = {0: Q4, 1: Q4 + n_cam / 8 / 2**30, 2: Q4 + 2 * n_cam / 8 / 2**30}
NOME_N = {0: 'Q4_K_M_imat_agora', 1: 'nível 1', 2: 'nível 2'}

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
T = ['# STRATA, C2: alocador pelo efeito medido na saída (Llama-3.1-8B, caminho F16)', '',
     f'GPU: {GPU}. KL contra o BF16 em {KLD_CHUNKS} trechos de 512 tokens de cada texto (wikitext-2 raw de teste; C4 en, validação 00000). '
     f'Níveis: 1 = P, 2 = D linha, bits do GPTQ (Hessianas com {HESS_CHUNKS} trechos, amortecimento {AMORTECE}). ΔKL medido em {DKL_CHUNKS} trechos de cada texto da medida, grupos de {BANDA} camadas. '
     f'Matrizes das camadas: {n_cam / 1e9:.2f} bilhões de pesos. Nível 0 = Q4_K_M com imatrix ({Q4:.3f} GiB); RAM do nível 1 {RAM_N[1]:.3f} GiB, do nível 2 {RAM_N[2]:.3f} GiB.', '',
     '## Níveis inteiros e a base', '', '| modelo | GiB | wikitext | C4 | bits a mais (wiki / C4) |', '|---|---:|---:|---:|---:|']
for k in (0, 1, 2):
    w, c = kld.get((NOME_N[k], 'wiki')), kld.get((NOME_N[k], 'c4'))
    if w and c:
        T.append(f'| {NOME_N[k]} | {RAM_N[k]:.3f} | {1000 * w[0]:.2f} ± {1000 * w[1]:.2f} | {1000 * c[0]:.2f} ± {1000 * c[1]:.2f} | '
                 f'{bits(RAM_N[k], w[0], "wiki"):+.3f} / {bits(RAM_N[k], c[0], "c4"):+.3f} |')
conf = all(abs(kld[('Q4_K_M_imat_agora', t)][0] - kld[('Q4_K_M_imat', t)][0])
           <= 2 * math.hypot(kld[('Q4_K_M_imat_agora', t)][1], kld[('Q4_K_M_imat', t)][1]) for t in BASE)
T += ['', f'Q4_K_M da frente (rodada do C1b): {1000 * kld[("Q4_K_M_imat", "wiki")][0]:.2f} / {1000 * kld[("Q4_K_M_imat", "c4")][0]:.2f}. '
      f'**Base igual (dentro de 2 barras de erro):** {"sim" if conf else "NÃO: a frente reaproveitada não vale para esta base"}', '',
      '## Misturas (KL ×1000; bits a mais contra a frente)', '',
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
T += ['', '## A soma prevê a medida? (KL do PyTorch, textos da medida)', '', '| orçamento | prevista | medida | erro |', '|---|---:|---:|---:|']
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
os.makedirs('/content/res', exist_ok=True)
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

nb = {"cells": celulas, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "L4"},
                                      "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                      "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(AQUI, RES + "_colab.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print(out)
