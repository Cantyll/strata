# E4: subir de nível com os planos chegando pela rede (Llama-3.2-1B, container)

O mesmo programa do E3 ([`strata_subir.cpp`](../strata_subir.cpp)), com os planos chegando a uma taxa fixa, uma rede
simulada (`--mbit`). Os bytes saem do arquivo local, e uma thread só os libera quando, à taxa dada, já teriam chegado
inteiros. A ordem é a dos arquivos de estrato: todos os planos do nível 1, depois os do nível 2 (121,6 MB cada no 1B).
Não houve rede de verdade, nem conferência de SHA-256, nem o formato `model-e01`/`e02` da distribuição: o que está
medido é o comportamento da geração enquanto os planos chegam. A geração começa no Q4_K_M já baixado, com 4 threads,
e a subida começa no token 16. Container de 4 núcleos virtuais, máquina compartilhada, 2026-10-23.

## O primeiro token

Os planos chegam depois do GGUF, então o primeiro token sai na hora em que sairia com o Q4_K_M sozinho. Isso vale por
construção e não precisou de medida. No 1B a 100 Mbit/s, o Q4_K_M (808 MB) chega em 64,6 s. O nível 2 fica completo
19,5 s depois, aos 84,1 s, enquanto o Q6_K (1 022 MB) baixado direto estaria pronto aos 81,7 s, 3% antes. É a mesma
conta da seção 2.1 para o 8B.

## Duas formas de preparar os planos

- **Thread de fundo** (E3): uma thread lê e converte cada matriz assim que ela chega. Com a geração em 4 threads, ela
  disputa os núcleos. A variante `--t-subida 3` libera um núcleo só enquanto a thread trabalha, e não resolve, porque
  as rajadas começam no meio de um token.
- **Cooperativa** (`--orcamento-ms 4`): a thread da rede só recebe os bytes. A conversão roda na própria geração,
  entre dois tokens, em pedaços, até gastar 4 ms. Cada plano entra direto como tipo S, e o Q6_K no nível 2 entra como
  `STRATA_Q6_K_S2`, um tipo novo, bit a bit igual ao P2 (conferido como no E3). Com o orçamento que sobra, o Q6_K é
  montado em P2 depois, o "polimento", que só melhora a velocidade. Os buffers dos níveis são reservados e tocados antes
  da geração (396 MB em 0,3 s no 1B). Numa VM, preparar em memória nova foi 10 a 20× mais lento por causa das faltas
  de página (0,47–0,78 contra 0,04 ns por peso).

## Resultado

Tempo de cada token em ms (com a preparação feita antes dele). "Normal" é a mediana dos 16 tokens antes da subida.
O controle é a mesma geração sem subida nenhuma, e mostra o ruído da máquina.

| forma | rede | nível 1 / nível 2 completos (ideal da rede) | P2 montado | p99 | máximo (em tokens normais) |
|---|---|---|---|---:|---:|
| controle, sem subida | | | | 95 · 70 | 4,27 · 1,68 |
| thread de fundo, 4 threads | 50 / 100 / 300 Mbit/s | na hora da rede | (montado direto) | 137 · 270 · 407 | 10,6 · 10,4 · 11,1 |
| thread de fundo, 3 threads quando ocupada | 50 / 100 / 300 | na hora da rede | (montado direto) | 100 · 97 · 104 | 8,4 · 4,4 · 6,0 |
| **cooperativa, 4 ms** | 50 Mbit/s | 19,5 / 39,7–40,3 s (19,5 / 38,9) | 41,7–42,3 s | 98 · 98 | 2,31 · 2,13 |
| **cooperativa, 4 ms** | 100 Mbit/s | 9,8 / 19,9–20,3 s (9,7 / 19,5) | 23,7–25,7 s | 84 · 93 | 1,85 · 2,18 |
| **cooperativa, 4 ms** | 300 Mbit/s | 3,3 / 7,4 s (3,2 / 6,5) | depois do fim da rodada | 95 · 91 | 2,01 · 2,35 |
| cooperativa, 4 ms | disco (sem limite) | 1,0 / 1,5 s | 16,3 s | 100 | 4,89 |

- **A subida acompanha a rede.** O nível 1 fica completo no instante em que o último byte chega. O nível 2 fica 0,4 a
  0,9 s depois da rede, e o P2 de 2 a 6 s depois.
- **No modo cooperativo, a geração não trava além do ruído da máquina.** O p99 (84–98 ms) e o máximo (1,9–2,4 tokens
  normais) ficaram dentro do que o controle sem subida mostrou (p99 70–95, máximo 1,7–4,3). A thread de fundo tem
  travadas sistemáticas: o p99 chega a 270–400 ms e o máximo a 10–11 tokens.
- **O critério "pausa < 1 token" não dá para medir nesta máquina.** O próprio controle, sem subida, teve um token de
  4,3 tokens normais. O que dá para afirmar é que a subida cooperativa não pôs travadas acima desse ruído. A medida fina
  fica para uma máquina estável (E6).
- **Velocidade durante a subida:** 15–16 tokens/s, contra 19–22 antes. A maior parte é o próprio nível ficando mais
  pesado (o nível 2 roda a ~17 tokens/s, E3), e o resto são os 4 ms por token.

Resumos de cada rodada (CSV por token em [`e4_csv/`](e4_csv/); `e4_*_t0`/`t3` = thread de fundo com 4 threads / 3
threads quando ocupada, `e4x_coop_*` = cooperativa, `e4x_ctrl_*` = controle):

```
e4_50_t0.csv: antes p50 46.2 ms | durante 610 tokens, p50 58.8, p99 136.6, máx 490.4 ms (máx = 10.61 tokens normais; pausa = 9.61) | depois p50 71.3 | trocas aplicadas 224 em 176 tokens, máx 40 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 20.3 s, nível 2 em 39.9 s; tokens/s durante a subida 15.7, antes 21.6
e4_50_t3.csv: antes p50 46.0 ms | durante 624 tokens, p50 58.6, p99 99.9, máx 386.3 ms (máx = 8.40 tokens normais; pausa = 7.40) | depois p50 70.0 | trocas aplicadas 224 em 182 tokens, máx 11 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 20.4 s, nível 2 em 39.8 s; tokens/s durante a subida 16.1, antes 21.8
e4_100_t0.csv: antes p50 47.5 ms | durante 296 tokens, p50 56.6, p99 270.1, máx 493.5 ms (máx = 10.39 tokens normais; pausa = 9.39) | depois p50 63.8 | trocas aplicadas 224 em 138 tokens, máx 4 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 10.6 s, nível 2 em 20.4 s; tokens/s durante a subida 15.2, antes 21.1
e4_100_t3.csv: antes p50 49.4 ms | durante 315 tokens, p50 57.9, p99 97.1, máx 218.2 ms (máx = 4.41 tokens normais; pausa = 3.41) | depois p50 64.1 | trocas aplicadas 224 em 140 tokens, máx 11 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 10.6 s, nível 2 em 20.4 s; tokens/s durante a subida 16.2, antes 20.2
e4_300_t0.csv: antes p50 52.1 ms | durante 93 tokens, p50 56.1, p99 406.9, máx 576.6 ms (máx = 11.08 tokens normais; pausa = 10.08) | depois p50 61.7 | trocas aplicadas 224 em 87 tokens, máx 12 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 4.2 s, nível 2 em 8.3 s; tokens/s durante a subida 12.5, antes 19.2
e4_300_t3.csv: antes p50 46.3 ms | durante 108 tokens, p50 59.4, p99 104.0, máx 277.5 ms (máx = 6.00 tokens normais; pausa = 5.00) | depois p50 63.4 | trocas aplicadas 224 em 78 tokens, máx 12 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 4.2 s, nível 2 em 8.1 s; tokens/s durante a subida 14.9, antes 21.6
e4x_coop_50_1.csv: antes p50 50.0 ms | durante 650 tokens, p50 60.9, p99 98.0, máx 115.6 ms (máx = 2.31 tokens normais; pausa = 1.31) | depois p50 72.3 | trocas aplicadas 240 em 155 tokens, máx 4641 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 20.4 s, nível 2 em 40.6 s; tokens/s durante a subida 15.7, antes 20.0
e4x_coop_50_2.csv: antes p50 50.7 ms | durante 644 tokens, p50 63.5, p99 98.4, máx 107.9 ms (máx = 2.13 tokens normais; pausa = 1.13) | depois p50 76.4 | trocas aplicadas 240 em 148 tokens, máx 6947 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 20.4 s, nível 2 em 41.2 s; tokens/s durante a subida 15.3, antes 19.7
e4x_coop_100_1.csv: antes p50 51.7 ms | durante 375 tokens, p50 63.0, p99 84.4, máx 95.7 ms (máx = 1.85 tokens normais; pausa = 0.85) | depois p50 63.9 | trocas aplicadas 240 em 104 tokens, máx 6435 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 10.6 s, nível 2 em 21.2 s; tokens/s durante a subida 15.9, antes 19.3
e4x_coop_100_2.csv: antes p50 45.6 ms | durante 388 tokens, p50 66.5, p99 93.4, máx 99.4 ms (máx = 2.18 tokens normais; pausa = 1.18) | depois p50 65.8 | trocas aplicadas 240 em 108 tokens, máx 4694 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 10.6 s, nível 2 em 20.7 s; tokens/s durante a subida 15.2, antes 22.0
e4x_coop_300_1.csv: antes p50 52.2 ms | durante 203 tokens, p50 66.6, p99 94.7, máx 105.1 ms (máx = 2.01 tokens normais; pausa = 1.01) | depois p50 80.6 | trocas aplicadas 235 em 79 tokens, máx 4764 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 4.2 s, nível 2 em 8.3 s; tokens/s durante a subida 15.1, antes 19.1
e4x_coop_300_2.csv: antes p50 51.4 ms | durante 216 tokens, p50 64.2, p99 90.9, máx 120.8 ms (máx = 2.35 tokens normais; pausa = 1.35) | depois p50 nan | trocas aplicadas 237 em 77 tokens, máx 5061 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 4.3 s, nível 2 em 8.5 s; tokens/s durante a subida 15.0, antes 19.4
e4x_coop_disco.csv: antes p50 50.4 ms | durante 227 tokens, p50 69.3, p99 99.7, máx 246.6 ms (máx = 4.89 tokens normais; pausa = 3.89) | depois p50 68.1 | trocas aplicadas 240 em 36 tokens, máx 8695 µs | níveis completos: 1 = True, 2 = True
    nível 1 completo em 1.9 s, nível 2 em 2.4 s; tokens/s durante a subida 13.9, antes 19.8
e4x_ctrl_1.csv: sem subida | p50 dos 16 primeiros 50.1 ms | p50 54.3, p99 95.3, máx 214.0 ms (máx = 4.27 tokens normais)
e4x_ctrl_2.csv: sem subida | p50 dos 16 primeiros 52.8 ms | p50 53.2, p99 69.9, máx 88.5 ms (máx = 1.68 tokens normais)

e4x_coop_100_1.log: subida: nível 1 completo em 9.76 s, nível 2 em 20.34 s (desde o token 16) Q6_K montados em P2 em 23.66 s 
e4x_coop_100_2.log: subida: nível 1 completo em 9.77 s, nível 2 em 19.87 s (desde o token 16) Q6_K montados em P2 em 25.69 s 
e4x_coop_300_1.log: subida: nível 1 completo em 3.26 s, nível 2 em 7.38 s (desde o token 16) Q6_K montados em P2 em -1.00 s 
e4x_coop_300_2.log: subida: nível 1 completo em 3.29 s, nível 2 em 7.41 s (desde o token 16) Q6_K montados em P2 em -1.00 s 
e4x_coop_50_1.log: subida: nível 1 completo em 19.48 s, nível 2 em 39.73 s (desde o token 16) Q6_K montados em P2 em 41.73 s 
e4x_coop_50_2.log: subida: nível 1 completo em 19.50 s, nível 2 em 40.30 s (desde o token 16) Q6_K montados em P2 em 42.26 s 
e4x_coop_disco.log: subida: nível 1 completo em 1.00 s, nível 2 em 1.54 s (desde o token 16) Q6_K montados em P2 em 16.33 s
```

## Limites

- Rede simulada: sem perda, sem variação de taxa, sem o custo da conferência dos hashes (SHA-256 de 243 MB custa ~0,5 s
  de uma thread, o que caberia no orçamento) e sem o formato de arquivo da distribuição. Ligar a leitura aos arquivos
  `model-e0k` com hash conferido por matriz é trabalho de integração, não de medida.
- 1B e 4 núcleos ruidosos. No 8B, o polimento do Q6_K leva ~7× mais (≈6 s de trabalho, ≈90 s de geração a 4 ms por
  token), e o tempo entre tokens também muda. Isso precisa ser medido lá.
