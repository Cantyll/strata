# E6 no 1B (container): subida de nível lendo os arquivos de estrato reais, com download de verdade

Llama-3.2-1B (unsloth), Q4_K_M do E3 como nível 0, CPU Xeon 2,1 GHz com 4 núcleos (container, sem GPU). É a parte do
E6 de [docs/08](../../../docs/08-precisao-sob-demanda.md) que dá para fazer no container. O 8B vai para o Colab
([`strata_8b_e6.ipynb`](../strata_8b_e6.ipynb)).

## O que mudou em relação ao E4

| | E4 | E6 |
|---|---|---|
| de onde vêm os planos | um arquivo de planos próprio (`strata_gguf_niveis.py --planos`) | `model-e01` e `model-e02` gravados pelo `strata-ref distribui` (B.10), os que iriam para o Hub |
| integridade | nenhuma | SHA-256 de cada plano conferido contra o índice antes de entrar no modelo |
| rede | simulada dentro do programa | servidor HTTP com pedidos de faixa e teto de banda (`tests/dist/servidor_faixas.py`) + baixador (`baixa_estratos.py --ate-nivel 2`, 8 conexões, pedaços de 8 MB) em processos à parte |
| controle | geração sem subida | geração **com o mesmo download** e sem subida (`--alvo 0`), na mesma faixa de tokens |

```
strata-ref distribui q4km.gguf dist model.safetensors                     # e00..e04 + índice (350 s no container)
python tests/demanda/estratos_idx.py dist/model.strata.index.json q4km.gguf dist/subida.idx
python tests/dist/servidor_faixas.py dist --porta 8800 --mbit 100 &
strata_subir gerar q4km.gguf baixa/subida.idx prompt.txt saida.csv -t 4 -n 900 --inicio 16 --alvo 2 --orcamento-ms 4 \
    --t-subida 3 --chat 1 --espera 30 --baixar "python3 tests/dist/baixa_estratos.py http://127.0.0.1:8800 baixa --ate-nivel 2"
```

## Os planos da distribuição são os mesmos

- Os planos de nível 1 e 2 dentro de `e01` e `e02` são **iguais byte a byte** aos do arquivo de planos do E3 nas 112
  matrizes das camadas (224 de 224). A ligação entre os nomes do HF e os do GGUF sai do stream `q` de cada matriz,
  que aponta para o offset do tensor no GGUF.
- O modo `conferir`, lendo de `e01`/`e02`, dá logits **iguais bit a bit** aos dos GGUFs carregados do zero, em 128
  posições e nos seis estados: 0, 1, 2, 2 com o Q6_K separado, 1 depois de descer e 0 depois de descer. Foram
  256 planos conferidos pelo SHA-256.
- Com um byte trocado num plano de `e01`, o programa para com
  `blk.1.ffn_gate.weight: o plano do nível 1 não bate com o SHA-256 do índice; não entra no modelo`.

## Subida com o download de verdade

Oito rodadas: 100 e 300 Mbit/s; geração com 4 threads o tempo todo (t0) ou com 3 enquanto o download corre (t3); duas
repetições. Os tempos contam a partir do token 16, quando o baixador é lançado. A faixa vai do primeiro ao último token
com troca; o controle usa a mesma faixa.

| teto | threads | rep. | download termina | nível 1 completo | nível 2 completo | Q6_K montado (P2) | ms/token antes | p99 subida / controle (ms) | pior token subida / controle (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | t0 | 1 | 27,6 s | 16,7 s | 29,4 s | 37,0 s | 47 | 169 / 193 | 477 / 419 |
| 100 | t0 | 2 | 26,5 s | 17,7 s | 28,2 s | 32,7 s | 43 | 162 / 130 | 382 / 284 |
| 300 | t0 | 1 | 10,4 s | 8,3 s | 11,6 s | 24,7 s | 45 | 280 / 163 | 439 / 336 |
| 300 | t0 | 2 | 10,7 s | 8,2 s | 12,9 s | 21,3 s | 46 | 196 / 1.814 | 486 / 2.373 |
| 100 | t3 | 1 | 25,4 s | 17,0 s | 26,6 s | 32,8 s | 57 | 90 / 68 | 172 / 263 |
| 100 | t3 | 2 | 25,1 s | 16,5 s | 25,6 s | 31,9 s | 55 | 82 / 71 | 92 / 85 |
| 300 | t3 | 1 | 8,6 s | 5,6 s | 10,2 s | 20,9 s | 53 | 75 / 70 | **796** / 97 |
| 300 | t3 | 2 | 8,8 s | 5,9 s | 9,9 s | 20,1 s | 55 | 83 / 73 | 105 / 114 |

Em todas as rodadas com subida, os 240 planos usados foram conferidos (112 do nível 1, 112 do nível 2 e o plano 1 dos
16 Q6_K, relido para a montagem em P2), e os dois níveis completaram.

- **O que trava aqui é o download, não a subida.** Com 4 threads, o controle (o mesmo download, sem subida) já tem
  p99 de 130 a 1.814 ms e pior token de 6 a 52 tokens normais. O servidor e o baixador são processos Python que
  disputam os 4 núcleos com as threads do ggml, que se esperam a cada operação (o mesmo efeito do E3).
- **Com um núcleo livre durante o download (t3), as travadas somem.**
  - O p99 fica em 75–90 ms com subida e 68–73 ms no controle (1,1 a 1,3× o controle).
  - O pior token fica em 1,7–3,0 tokens normais em três das quatro rodadas.
  - A exceção foi um token de 796 ms na rodada 300/t3/1. Ele não vem do nosso trabalho: o preparo daquele token
    gastou 4,0 ms, dentro do orçamento, e o resto foi o `llama_decode`. Aconteceu perto do fim do download.
  - O custo é rodar ~20% mais devagar enquanto o download corre (55 contra 45 ms/token no 1B).
- **O nível 1 completa antes do fim do download**, porque o `e01` vem inteiro antes do `e02`. **O nível 2 completa
  de 0,8 a 1,5 s depois do último byte**, um pouco mais que no E4 (0,4–0,9 s), por causa da conferência do SHA-256.
  Ela é feita junto com a conversão, nos mesmos 4 ms entre tokens, e custa 0,09 ns/peso: a preparação dos planos foi
  de 0,04–0,06 ns/peso para 0,14 ns/peso.
- **O primeiro plano demora a chegar:** a primeira troca acontece ~8 s depois do início a 100 Mbit/s. O baixador pede
  16 pedaços ao mesmo tempo, e eles dividem a banda e chegam juntos. Isso não muda quando cada nível completa, mas
  numa subida gradual seria melhor pedir em ordem.
- **Bytes que a subida não usa:** `e01` e `e02` também trazem os planos do `token_embd`, que não sobe de nível (E2).
  No 1B são 33 MB de 154 MB por arquivo (21%, porque o 1B usa o embedding também como saída). No 8B seriam 66 MB de
  embedding mais 66 MB de `output` por arquivo, ~13%. O baixador poderia pedir só as faixas das matrizes das camadas.

## Limites

- O 1B não serve como demonstração: o modelo base repete o mesmo parágrafo
  ([resposta](e6_csv/m100_t3_a2_2_resposta.txt)), com e sem troca. O texto é contínuo através das marcas, mas a
  coerência só dá para julgar no 8B Instruct.
- A rede é um teto de banda local, sem perda e sem latência variável. O servidor roda na mesma máquina e disputa CPU,
  o que piora os números em relação a um servidor remoto.
- São 4 núcleos numa VM, com ruído grande entre rodadas (o controle 300/t0/2 teve um token de 2,4 s).
