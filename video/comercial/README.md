# Vídeo comercial do STRATA

Apresentação de produto, 9:16, 3 min 51 s, com tudo o que foi construído e medido do PR #1 ao PR #11, guiada pelo **Bit**, um personagem feito dos cinco estratos do logo.

| Arquivo | O que é |
|---|---|
| [`strata_comercial_9x16.mp4`](strata_comercial_9x16.mp4) | o vídeo (1080 × 1920, 30 q/s, H.264 crf 27 + AAC, −16 LUFS) |
| [`strata_comercial_roteiro.md`](strata_comercial_roteiro.md) | as 13 cenas, legendas e a fonte de cada número |
| [`strata_comercial_quadros.png`](strata_comercial_quadros.png) | um quadro por cena |
| `fonte/anim.html` | a animação: `draw(t)` determinística, padrão v2 de `video/README.md` (PR #11) |
| `fonte/narr/p01.mp3` … `p13.mp3` | a narração, um parágrafo por pedido ao ElevenLabs |
| `fonte/monta_tempos.py` | junta os parágrafos com 0,8 s de pausa e gera `narracao.mp3`, `tempos.js` (cenas e legendas) e `boca.js` (a boca do Bit segue o volume da voz) |
| `fonte/palavras.json` | tempos por palavra do faster-whisper `small` |
| `fonte/render.mjs`, `fonte/monta.sh` | cópia do pipeline de `video/padrao/` |

Para refazer, dentro de `fonte/`: `python3 monta_tempos.py`, depois `./monta.sh anim.html narracao.mp3 231 saida.mp4` (node + Playwright e ffmpeg; ~9 min em 4 núcleos) e recomprimir com crf 27 para ficar abaixo de 25 MB.

Não há cena de veredito: pela regra do projeto, a conclusão só é gravada depois do merge.
