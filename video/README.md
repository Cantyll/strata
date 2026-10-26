# Padrão de vídeo do STRATA

Regra do projeto: toda sessão nova e todo PR novo ganham uma animação explicativa com narração.
Este arquivo fixa o formato aprovado no PR #11 (versão 2, "apresentação de produto"). O exemplo
completo está em [`padrao/exemplo-pr11.html`](padrao/exemplo-pr11.html); para um vídeo novo, copie-o
e troque as cenas.

## Formato

| Item | Valor |
|---|---|
| Proporção | 9:16, 1080 × 1920, 30 quadros/s |
| Duração | 60 a 180 s (nem menos, nem mais) |
| Conteúdo | didático: uma ideia por cena, números grandes, frases curtas na tela |
| Narração | ElevenLabs, voz "Henrique - Clear and Knowledgeable" (pt-BR), modelo `eleven_v4` |
| Legenda | uma frase por vez, embaixo, com os tempos medidos da narração |
| Saída | H.264 crf 22 + AAC 160 k, `+faststart`, normalizado a −16 LUFS |

## Cores

Fundo grafite com um único degradê de marca (azul → ciano → menta). Dourado só para o Q4_K_M,
que é o "arquivo que você já tem"; não usar dourado para mais nada.

| Token | Cor | Uso |
|---|---|---|
| `--bg` | `#07080C` | fundo |
| `--ink` / `--mute` / `--dim` | `#F5F6FA` / `#9AA0B4` / `#5A6074` | texto principal, secundário, apagado |
| degradê | `#7B61FF → #00C2FF → #2EE6A6` | títulos de destaque, logo, barras do STRATA |
| `--q4` | `#F5B83D` | o GGUF Q4_K_M (estrato 0) |
| `--n1` `--n2` `--n3` | `#5B6CFF` `#3B9BFF` `#22C6F0` | planos de 1 bit (níveis 1, 2, 3) |
| `--fe` / `--ok` | `#2EE6A6` | fecho até o BF16 exato; "confere" |
| `--warn` | `#FFB547` | limites, ressalvas |
| `--bad` | `#FF4D6D` | recusado, perde |

## Tipografia

- Títulos: **Inter Display 800**, grandes (110–180 px para o número principal da cena).
- Texto: Inter 500–600.
- Números e nomes de arquivo: JetBrains Mono. **Não vem instalada no contêiner** (cai em serif
  sem aviso); instale ou use DejaVu Sans Mono.

## Movimento

- Cada parágrafo da narração é uma cena. Entre cenas, **transição em cubo 3D**: eixo Y nas pares,
  X nas ímpares. Meia aresta do cubo = W/2 para o eixo Y e H/2 para o eixo X; com outro valor as
  faces se sobrepõem.
- No meio do giro a câmera recua um pouco; dentro da cena há uma **deriva lenta de câmera**.
- O título da cena nova começa a entrar **0,45 s antes do corte**, para a face não chegar vazia.
- Elementos entram com `entra()`: viram de rotX/rotY e vêm de longe no eixo Z.
- Cartões de lista (limites, passos) giram para a frente como um carrossel.
- Fundo vivo: holofote, névoa e chão quadriculado em perspectiva.
- Tudo é função do tempo: `draw(t)` desenha o quadro do instante `t` sem estado escondido. É isso que
  permite renderizar em partes paralelas e repetir um quadro exato.

## Honestidade

O vídeo segue as mesmas regras dos documentos: só números medidos, e uma cena de limites (onde
o STRATA perde ou ainda não foi testado). Conclusão de qualidade só com resultado no 8B.

## Como fazer

1. **Roteiro**: um parágrafo por cena, 150–170 palavras por minuto. No texto do TTS, escreva
   siglas pela pronúncia ("quê quatro ká ême", "bê éfe dezesseis", "lhama cê pê pê").
2. **Narração**: `creative_generate_speech` do conector ElevenLabs (voz acima). Em erro 429, tente
   de novo uma vez.
3. **Tempos**: transcreva o mp3 com tempos por palavra (Scribe do ElevenLabs ou faster-whisper
   `small` local). Preencha `tempos.js`: `S` = início de cada cena, `DUR`, `CAPS` = legendas.
4. **Página**: copie `padrao/exemplo-pr11.html`, aponte o `<script src>` para o seu `tempos.js` e
   troque o conteúdo das cenas. Confira quadros soltos:
   `PAGINA=anim.html node padrao/render.mjs stills 1,20,40`.
5. **Render**: `padrao/monta.sh anim.html narracao.mp3 <DUR> saida.mp4` (4 partes em paralelo,
   narração entra 0,5 s depois, loudnorm, crf 22). Precisa de node + Playwright e ffmpeg.
6. **Entrega**: mp4, roteiro e quadros-chave na pasta do projeto; fonte (html, tempos, mp3) junto.

## Lista de conferência

- [ ] 9:16, entre 60 e 180 s
- [ ] Cada cena tem uma ideia e um número ou imagem central
- [ ] Siglas pronunciadas certo (ouça o começo de cada cena)
- [ ] Nenhum texto sai da tela ou fica sob a legenda
- [ ] Transições em cubo sem faces sobrepostas
- [ ] Cena de limites presente; números batem com `docs/05-resultados.md`
- [ ] Arquivo final abaixo de ~25 MB
