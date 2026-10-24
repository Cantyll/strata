# Junta os 13 parágrafos da narração (narr/pNN.mp3) com pausas, mede as legendas pelas palavras
# do faster-whisper (palavras.json) e grava narracao.mp3, tempos.js e boca.js (envelope da voz, 30 q/s).
import json, re, subprocess, numpy as np
PAUSA = 0.8
P = json.load(open('palavras.json'))
# legenda exibida, frase-âncora (começo da frase como o whisper ouviu)
CAPS = [
 [("Oi! Eu sou o Bit, e vou ser o seu guia pelo STRATA.", "oi eu"),
  ("Em alguns minutos, você vai ver o que construímos,", "em alguns"),
  ("o que medimos, e onde ainda perdemos.", "o que medimos"),
  ("Vem comigo.", "vem comigo")],
 [("Modelos de IA são enormes, e a internet guarda cada um várias vezes.", "modelos de"),
  ("O Llama de 8 bilhões de parâmetros aparece no Hugging Face", "o lhama"),
  ("como o original e como três versões comprimidas:", "como o original"),
  ("33 GB no total.", "33 gb"),
  ("Num único repositório, achamos 29 versões do mesmo modelo.", "num único")],
 [("A ideia do STRATA é guardar só a surpresa:", "a ideia"),
  ("só o que ainda não dá para prever.", "só o que"),
  ("Os bits de cada peso viram camadas,", "os bits"),
  ("do mais importante ao menos importante.", "do mais"),
  ("Ler o começo já dá um modelo.", "ler o começo"),
  ("Ler tudo devolve o original, bit a bit, com a mesma impressão digital.", "ler tudo")],
 [("Primeiro passo: o arquivo sem perdas.", "primeiro passo"),
  ("O expoente de cada peso ocupa 8 bits,", "o expoente"),
  ("mas carrega só 2,6 de informação.", "mas carrega"),
  ("Um codificador chamado rANS guarda só isso.", "um codificador"),
  ("No Llama de 1 bilhão, o arquivo ficou com 66% do tamanho original,", "no lama"),
  ("e voltou idêntico.", "e voltou")],
 [("Depois, a pergunta difícil:", "depois a"),
  ("um pedaço do arquivo compete com os formatos comprimidos que todo mundo usa,", "um pedaço"),
  ("os k-quants do llama.cpp?", "os k-quants"),
  ("A resposta foi não.", "a resposta"),
  ("O Q4_K_M errou 3 vezes menos que o nosso corte do mesmo tamanho.", "o q4km"),
  ("A diferença era estrutural.", "a diferença")],
 [("Então mudamos a base.", "então mudamos"),
  ("No formato 0.3, a camada 0 é o próprio Q4_K_M, byte a byte.", "no formato"),
  ("Cada nível acima custa 1 bit por peso", "cada nível"),
  ("e divide o degrau da régua ao meio.", "e divide"),
  ("A última camada, o fecho, completa o original exato.", "a última"),
  ("Tudo em cerca de 11 bits por peso.", "tudo em")],
 [("Um formato novo precisa ser seguro.", "um formato"),
  ("Atacamos o leitor com 30 mil arquivos adulterados.", "atacamos"),
  ("No começo, quase 1 em cada 5 passava sem erro, com pesos errados.", "no começo"),
  ("Reescrevemos o leitor em Rust, com uma impressão digital para cada pedaço.", "rescrevemos"),
  ("Hoje, nenhum passa em silêncio:", "hoje nenhum"),
  ("mudou um byte, o arquivo é recusado.", "mudou um")],
 [("Modelos grandes chegam em vários arquivos,", "modelos grandes"),
  ("e cada um volta com a mesma impressão digital do Hugging Face.", "e cada um"),
  ("E não é preciso esperar o download acabar:", "e não é"),
  ("o STRATA monta o modelo enquanto baixa.", "os trata"),
  ("No Llama de 8 bilhões, a 100 megabits por segundo,", "no lhama"),
  ("o original fica pronto em 8 minutos e meio, contra 21 baixando direto.", "o original fica")],
 [("E os níveis do meio, rodam?", "e os níveis"),
  ("Criamos tipos novos dentro do llama.cpp, no processador e na placa de vídeo.", "criamos"),
  ("Na placa de vídeo, o nível 1 gera texto na velocidade do Q5_K_M.", "na placa de vídeo o"),
  ("Mas, na mesma memória, a qualidade ainda fica atrás dos k-quants com imatrix:", "mas na mesma"),
  ("de 1 a 2 décimos de bit por peso.", "de 1 a")],
 [("Onde o STRATA ganha de verdade é no armazenamento.", "onde o"),
  ("O Q4_K_M do Hub fica intacto, como camada zero.", "o que 4km"),
  ("No Llama de 8 bilhões, 4 arquivos com 33 GB viram um conjunto de 11:", "no llama"),
  ("66% a menos.", "66"),
  ("No de 70 bilhões, de 291 para 98 GB,", "no de 70"),
  ("com os 30 arquivos de volta, bit a bit.", "com os 30")],
 [("E quem já tem o Q4_K_M?", "e quem"),
  ("Baixa só 6 GB de camadas para ter o original, em vez de 16.", "baixa só")],
 [("Agora, os limites, sem enfeite.", "agora os"),
  ("Os níveis do meio ainda perdem para os k-quants", "os níveis"),
  ("e precisam de um llama.cpp modificado.", "e precisam"),
  ("Em rede rápida, acima de uns 670 megabits,", "em rede"),
  ("baixar o original direto ainda é mais rápido.", "baixar o"),
  ("E testamos só a família Llama, com a rede simulada.", "e testamos")],
 [("O próximo passo é publicar as camadas no Hub de verdade", "o próximo"),
  ("e medir com outros modelos.", "e medir"),
  ("Eu sou o Bit. Até a próxima etapa do STRATA.", "eu sou")],
]
norm = lambda w: re.sub(r'[^\w%-]', '', w.lower())
def dur(f):  # duração pelas amostras decodificadas (a do cabeçalho do mp3 inclui o enchimento do codificador)
    raw = subprocess.run(['ffmpeg','-v','error','-i',f,'-f','s16le','-ac','1','-ar','44100','-'],capture_output=True,check=True).stdout
    return len(raw) / 2 / 44100
S, caps, t = [], [], 0.0
pecas = []
for i, lista in enumerate(CAPS):
    f = f'p{i+1:02d}.mp3'; ws = P[f]; d = dur('narr/' + f)
    S.append(round(t, 2))
    W = [norm(w) for _, _, w in ws]
    # busca a âncora no texto colado (sem espaços), para não depender de como o whisper parte as palavras
    junto, dono = '', []
    for j, w in enumerate(W): junto += w; dono += [j] * len(w)
    idx, pos = [], 0
    for txt, anc in lista:
        p = junto.find(norm(anc.replace(' ', '')), pos)
        if p < 0: raise SystemExit(f'âncora "{anc}" não achada em {f}')
        idx.append(dono[p]); pos = p + 1
    for j, (txt, _) in enumerate(lista):
        s = ws[idx[j]][0]; e = ws[(idx[j+1] if j + 1 < len(idx) else len(ws)) - 1][1]
        caps.append([round(t + s, 2), round(t + e, 2), txt])
    pecas.append(('narr/' + f, d)); t += d + PAUSA
DUR = round(t - PAUSA + 1.5, 2)
# narração única com as pausas
lst = []
for k, (f, d) in enumerate(pecas):
    lst.append(f)
inputs = sum([['-i', f] for f in lst], [])
fc = ''.join(f'[{k}:a]aresample=44100,apad=pad_dur={PAUSA}[a{k}];' for k in range(len(lst)))
fc += ''.join(f'[a{k}]' for k in range(len(lst))) + f'concat=n={len(lst)}:v=0:a=1[o]'
subprocess.run(['ffmpeg','-y','-v','error',*inputs,'-filter_complex',fc,'-map','[o]','-c:a','libmp3lame','-b:a','192k','narracao.mp3'],check=True)
# confere: S deve bater com a soma real das peças com pausa
# envelope para a boca do Bit: RMS em janelas de 1/30 s, normalizado
raw = subprocess.run(['ffmpeg','-v','error','-i','narracao.mp3','-f','f32le','-ac','1','-ar','24000','-'],capture_output=True,check=True).stdout
a = np.frombuffer(raw, np.float32); hop = 800
n = len(a) // hop
rms = np.sqrt((a[:n*hop].reshape(n, hop) ** 2).mean(1))
ref = np.percentile(rms[rms > 1e-3], 99)
env = np.clip(rms / ref, 0, 1)
open('boca.js', 'w').write('// envelope da narração (0–1), 30 por segundo, para a boca do Bit\nconst BOCA = [' + ','.join(f'{x:.2f}'.rstrip('0').rstrip('.') or '0' for x in env) + '];\n')
with open('tempos.js', 'w') as o:
    o.write('// Tempos da narração (s, desde o início do áudio): parágrafos colados com pausa de %.1f s,\n// legendas pelas palavras do faster-whisper "small" (monta_tempos.py).\n' % PAUSA)
    o.write('const S = ' + json.dumps(S) + ';\nconst DUR = %s;\nconst CAPS = [\n' % DUR)
    for c in caps: o.write('  ' + json.dumps(c, ensure_ascii=False) + ',\n')
    o.write('];\n')
print('S', S); print('DUR', DUR, 'narracao', dur('narracao.mp3'))
