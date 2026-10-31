# Junta os 12 parágrafos da narração (narr/pNN.mp3) com pausas, mede as legendas pelas palavras
# do faster-whisper (palavras.json) e grava narracao.mp3, tempos.js e boca.js (envelope da voz, 30 q/s).
import json, re, subprocess, numpy as np
PAUSA = 0.8
P = json.load(open('palavras.json'))
# legenda exibida, frase-âncora (começo da frase como o whisper ouviu)
CAPS = [
 [("Oi! Eu sou o Bit.", "oi eu"), ("Juntei quatro pesquisas sobre o custo de inferência,", "juntei"),
  ("ou seja, de cada resposta de uma IA, num resumo só,", "ou seja"), ("com as notícias boas e as ruins.", "com as")],
 [("A regra de tudo:", "a regra"), ("para cada palavra, o computador lê os pesos inteiros da memória.", "para cada"),
  ("Nas máquinas populares, o llama.cpp já usa até 85% dessa estrada.", "nas máquinas"), ("Só bytes a menos dão ganho grande.", "só bytes")],
 [("A notícia ruim:", "a notícia"), ("com a mesma qualidade, o formato com treliça, o EXL3,", "com a mesma"),
  ("gasta cerca de 1 bit a menos por peso que os k-quants.", "gasta"), ("E os níveis do STRATA só empatam com os k-quants.", "e os níveis")],
 [("Letra miúda: ele só roda em placa Nvidia.", "letra"), ("E na CPU, o alvo do nosso MVP, ninguém mediu.", "e na cpu"),
  ("Também quebra a compatibilidade com o Q4_K_M do Hub.", "também")],
 [("No sem perda, todos empatam perto de dois terços do original.", "no sem"), ("E o nosso decodificador faz uns 100 MB por segundo,", "e o nosso"),
  ("70 vezes abaixo de um SSD rápido.", "70")],
 [("Duas pesquisas tiveram a mesma ideia:", "duas"), ("o Q4_K_M como rascunho, e o resto buscado só na conferência.", "o q4km"),
  ("Fizemos a conta: num modelo de 14 bilhões, em 12 GB,", "fizemos"), ("uns 15 tokens por segundo.", "uns 15"), ("O que já existe faz de 13 a 17. Empate.", "o que já")],
 [("Ir além da memória é coisa dos modelos de especialistas, os MoE:", "ir além"), ("um de 397 bilhões roda a mais de 4 tokens por segundo", "1 de 397"),
  ("num notebook de 48 GB.", "num notebook"), ("Ali, quase metade dos bytes a menos rendeu só 32%.", "ali"), ("Uma leitura mais esperta do SSD rendeu 2,5 vezes.", "uma leitura")],
 [("Para esses modelos, a lacuna mais clara é um cache por plano de bits.", "para esses"), ("Só que o STRATA ainda não lê esse tipo de modelo,", "só que"),
  ("e o ganho só aparece quando um especialista sobe de nível.", "e o ganho"), ("Um teste de horas mede isso antes.", "um teste")],
 [("No armazenamento, o rival mais forte é a receita:", "no armazenamento"), ("o original e as instruções que regeneram cada arquivo byte a byte.", "o original"),
  ("Ela já ganha do STRATA para quem hospeda.", "ela já"), ("O nosso lado é o de quem baixa:", "o nosso"), ("com o Q4_K_M, 6 GB em vez de 16.", "com o que")],
 [("A fila nova:", "a fila"), ("primeiro, deduplicação do Hugging Face, STRATA e receita, no mesmo conjunto.", "primeiro"),
  ("Segundo, com autorização, treliça contra k-quants na CPU.", "segundo"), ("Terceiro, uma treliça em camadas, no papel.", "terceiro"), ("Tudo aqui, sem placa de vídeo.", "tudo aqui")],
 [("A conclusão, sem enfeite:", "a conclusão"), ("o STRATA tem um lugar próprio,", "o strata"), ("um arquivo com várias precisões e volta exata ao original.", "um arquivo"),
  ("Mas qualidade por bit é da treliça,", "mas qualidade"), ("e ir além da memória é dos MoE.", "e ir"), ("Nas duas frentes, ainda falta a nossa peça.", "nas duas")],
 [("Eu sou o Bit. Até a próxima etapa do STRATA.", "eu sou")],
]
norm = lambda w: re.sub(r'[^\w%-]', '', w.lower())
def dur(f):  # duração pelas amostras decodificadas (a do cabeçalho do mp3 inclui o enchimento do codificador)
    raw = subprocess.run(['ffmpeg','-v','error','-i',f,'-f','s16le','-ac','1','-ar','44100','-'],capture_output=True,check=True).stdout
    return len(raw) / 2 / 44100
S, caps, t, PAL = [], [], 0.0, []
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
    PAL.append([[round(t + s, 2), round(t + e, 2), norm(w)] for s, e, w in ws])
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
    o.write('// palavras de cada cena (início, fim, palavra normalizada), para ancorar as animações\nconst PAL = ' + json.dumps(PAL, ensure_ascii=False) + ';\n')
print('S', S); print('DUR', DUR, 'narracao', dur('narracao.mp3'))
