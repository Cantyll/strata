# Vídeo de síntese das pesquisas (docs/12 a 15, cruzadas no docs/16)

Vídeo único pedido pelo Cantyll em 2026-10-31 para resumir as quatro pesquisas. Gravado completo,
com conclusão. 9:16, ~175 s, padrão v2 com o Bit e o quadro final.

Narração: ElevenLabs, voz "Henrique - Clear and Knowledgeable", `eleven_v4`, um parágrafo por pedido
(12 pedidos, ~2.400 créditos). Texto do TTS em `fonte/narracao_tts.txt`. Para o R retroflexo, o texto
evita palavras terminadas em -ar, -er e -ir no fim de frase (nenhuma frase termina assim); não foi
testada a grafia com acento.

| # | cena | o que mostra | origem |
|---:|---|---|---|
| 0 | Abertura | Bit; os quatro documentos | — |
| 1 | A regra de tudo | % do teto de banda que o llama.cpp já usa (M1 80, 3060 82, 4060 Ti 85, 5090 64, M2 Ultra 45) | doc 14 §1 |
| 2 | Um bit atrás | KL × bits: trellis (EXL3) contra k-quants; níveis do STRATA sobre a curva dos k-quants; ≈1 bit no tamanho do Q4_K_M | doc 12 §4 |
| 3 | Letra miúda | só CUDA; ninguém mediu na CPU; quebra o Q4_K_M do Hub como base | doc 12 G2, docs/16 K1 e K7 |
| 4 | Sem perda | ZipNN 66,4%, DFloat11 ~69%, STRATA 0.2 66,2%; decodificador ~0,1 GB/s contra SSD 7 e PCIe 25 | doc 12 §6, docs/16 K3 |
| 5 | Especulativa com resíduo | 14B em 12 GB: Q4 33, `-ngl` Q6_K 13–17, especulativa 15 tok/s (conta) | docs/16 §4.2 |
| 6 | Além da memória | 397B em 48 GB a 4,36 tok/s; 4→2 bits só +32%; leitura melhor 2,5× | doc 14 §2, docs/16 K2 |
| 7 | Cache por plano | ideia; STRATA não lê MoE; ganho só na subida; pré-teste de horas | doc 15 L1, docs/16 §4.3 |
| 8 | O rival: a receita | 33,31 / 11,18 / ~10,6 GB; quem tem o Q4_K_M baixa 6,25 em vez de 16,06 | docs/16 §4.5 |
| 9 | A fila nova | Xet × STRATA × receita; trellis na CPU (autorização); trellis em camadas; especulativa rebaixada | docs/16 §5 e §8 |
| 10 | Conclusão | ✓ lugar próprio; ✕ qualidade por bit é da treliça; ✕ além da memória é dos MoE | docs/16 §7 |
| 11 | Despedida | quadro final, logo + Bit acenando | padrão |

Como refazer: `fonte/monta_tempos.py` (precisa de `palavras.json` do faster-whisper "small"), depois
`fonte/monta.sh anim.html narracao.mp3 175.4 ../strata_sintese_9x16.mp4`.
Render final: monta.sh com crf 22 deu 34 MB; recomprimido com crf 25 (23 MB): ffmpeg -i saida.mp4 -c:v libx264 -preset slow -crf 25 -c:a copy -movflags +faststart
