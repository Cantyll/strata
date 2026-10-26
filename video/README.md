# Padrão de vídeo do STRATA

Regra do projeto: toda sessão nova e todo PR novo ganham uma animação explicativa com narração.
Este arquivo fixa o formato aprovado no PR #11 (versão 2, "apresentação de produto"). O exemplo
completo está em [`padrao/exemplo-pr11.html`](padrao/exemplo-pr11.html); para um vídeo novo, copie-o
e troque as cenas. Desde o vídeo comercial (PR #14), todo vídeo também tem o **Bit** como guia e
termina no **quadro final** (logo + Bit acenando); o exemplo completo com os dois é
[`comercial/fonte/anim.html`](comercial/fonte/anim.html).

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

## O Bit, o guia (obrigatório em todo vídeo)

Regra do Cantyll (24/10/2026): o Bit é o personagem padrão e aparece em todo vídeo, interagindo
com as cenas. O código é um só, [`padrao/bit.js`](padrao/bit.js); não redesenhe o Bit à mão.

**Como ele é** (não mudar sem pedido do Cantyll):

- **Corpo:** cinco estratos arredondados empilhados, o logo de pé. De cima para baixo: menta
  `#2EE6A6` (fecho), `#22C6F0`, `#3B9BFF`, `#5B6CFF` (níveis 3, 2, 1) e dourado `#F5B83D` na base,
  porque a base é o Q4_K_M. Larguras 96, 128, 140, 128, 104 px na escala 1, cada estrato com brilho
  da própria cor e um reflexo branco em cima.
- **Visor:** retângulo escuro arredondado sobre os três estratos do meio, com borda clara fina.
- **Olhos:** duas elipses brancas com brilho ciano. Piscam sozinhos a cada ~3–4,5 s e olham para o
  que ele aponta ou para onde ele voa.
- **Boca:** pílula menta que abre com o volume da narração (`boca.js`, 30 valores por segundo).
- **Antena:** haste ciano que balança, com uma bolinha menta que pulsa.
- **Mãos:** duas esferas azuis soltas dos lados. Apontam com um feixe pontilhado menta que termina
  num anel pulsando sobre o alvo, e a direita acena.
- **Movimento:** flutua (balanço leve), voa em arco entre posições, inclina na direção do voo e
  estica um pouco na vertical. Pode abrir os estratos do corpo (`desenha(a, separa)`) quando a cena
  fala de camadas.

**Como ele entra no vídeo:**

- Fica num `<canvas id="guia">` por cima do cubo e por baixo da legenda, então atravessa as
  transições sem girar com as faces: é o fio que liga as cenas.
- Na abertura, entra grande no centro (escala ~1,9), acena e se apresenta; depois vai para o
  corredor da direita (x ≈ 965, escala ~0,74). O conteúdo das cenas fica em x 80–860 (720 px de
  título, 780 px de cartões), para o Bit nunca cobrir texto.
- Em cada cena, desce pelo corredor até o elemento de que a narração fala e aponta para ele
  (chaves `BK` e gestos `GS`, ancorados nos tempos por palavra).
- A boca vem de `boca.js`, gerado do mp3 da narração (ver
  [`comercial/fonte/monta_tempos.py`](comercial/fonte/monta_tempos.py)).

## Quadro final (obrigatório em todo vídeo)

Todo vídeo termina igual: o logo de cinco estratos montando, "STRATA" e "um arquivo, várias
precisões" no centro (`Bit.final`), e o Bit descendo para baixo do logo e acenando
(`Bit.despedida`), na frase de despedida do narrador ("Eu sou o Bit. Até a próxima etapa do
STRATA."). Esse quadro não é a cena de veredito: ele entra em todo vídeo, inclusive antes do merge.

```js
const guia = Bit.cria($('guia'), { chaves: BK, gestos: GS, boca: BOCA });
BK.push(...Bit.despedida(t0).chaves); GS.push(...Bit.despedida(t0).gestos);
const fim = Bit.final(ultimaCena.p);
// em draw(a): const k = fim.anima(a, t0); (apague o resto da cena com 1 - k); guia.desenha(a);
```

## Honestidade

O vídeo segue as mesmas regras dos documentos: só números medidos, e uma cena de limites (onde
o STRATA perde ou ainda não foi testado). Conclusão de qualidade só com resultado no 8B.

## Como fazer

1. **Roteiro**: um parágrafo por cena, 150–170 palavras por minuto. No texto do TTS, escreva
   siglas pela pronúncia ("quê quatro ká ême", "bê éfe dezesseis", "lhama cê pê pê").
   O narrador apresenta o Bit na abertura e se despede como Bit no fim.
   **R no fim da palavra** (pedido do Cantyll, 24/10/2026): na voz Henrique, o R retroflexo fica
   forte demais em palavras terminadas em -ar, -er e -ir ("guardar", "prever", "medir"). Nos
   próximos textos de TTS, evite essas palavras no fim de frase e teste a grafia sem o R, com acento
   ("guardá", "prevê", "medí"). Ainda não foi testado: antes da narração inteira, gere só uma frase
   nas duas grafias (~150 créditos), ouça e use a melhor. Se nenhuma servir, teste outra voz pt-BR.
2. **Narração**: `creative_generate_speech` do conector ElevenLabs (voz acima), um parágrafo por
   pedido, com `generations_count` 1. Em erro 429, tente de novo uma vez. Colar os parágrafos com
   pausa fixa (`monta_tempos.py`) deixa o início de cada cena exato.
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
- [ ] O Bit aparece do começo ao fim, de `padrao/bit.js`, e nunca cobre texto
- [ ] Termina no quadro final: logo + Bit acenando
- [ ] Sem R forte no fim de -ar, -er, -ir (ouça as frases que terminam assim)
