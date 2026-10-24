# junta antigo × novo, Llama-3.1-8B, contêiner de 4 núcleos (2026-10-23)

Mesmos arquivos (GGUF do bartowski + estratos), cache de disco quente, sem `--tudo`, duas rodadas de cada, alternadas:

| versão | rodada 1 | rodada 2 |
|---|---:|---:|
| antigo (`070e992`, confere o GGUF inteiro sempre) | 539 s | 514 s |
| novo (`fda03c8`, em fila) | 284 s | 284 s |

Fases do novo (somadas por fase; correm em paralelo): leitura 11,5 s · hash dos streams 43,2 s · decodificação 216,7 s · bytes 12,9 s · hash da saída 111,8 s.
O SHA-256 em software roda a ~185 MB/s neste processador (sem extensão SHA).

`teste_grande_novo.md`: o teste completo refeito com o código novo; índice e arquivos de estrato byte a byte iguais aos da versão antiga.
