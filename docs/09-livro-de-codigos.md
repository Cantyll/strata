# Livro de códigos aninhado: o estrato 0 também vira livro

Plano escrito em 2026-10-23, antes de rodar qualquer medida. Pergunta: se o estrato 0 deixar de ser a grade uniforme do Q4_K_M e passar a ser um livro de códigos não uniforme, ajustado com a imatrix, e cada bit a mais dividir cada valor do livro em dois (a ideia do Any-Precision LLM, [pesquisa](07-quantizacao-aninhada.md)), o preço de aninhar some?

## O que já foi tentado e não se repete aqui

Quatro vertentes mexeram só nos níveis de cima e deixaram o nível 0 como o Q4_K_M ([resultados, seções 28 e 30](05-resultados.md#30-sub-células-por-tensor-no-llama-31-8b-tipos-d-na-gpu)). A melhor delas foi a dos tipos D, com deslocamentos de Lloyd por tensor sobre a grade intacta do Q4_K_M. No Llama-3.1-8B, ela ficou em +0,23 bit/peso no nível 1 e +0,11 no nível 2 contra os k-quants com imatrix no mesmo tamanho (KL ×1000 9,27 e 3,40 contra 7,13 e 3,09 interpolados). O que é novo aqui:

- **O nível 0 também é livro.** O livro não herda as 16 marcas uniformes do Q4_K.
- **Aninhamento por divisão.** O código do nível k é 2 × código do nível k − 1 + 1 bit, e o filho pode cair em qualquer ponto dentro da célula do pai.
- **Custo contado em bits por peso,** já com escalas e tabelas.
- **O estrato 0 deixa de ser igual ao Q4_K_M do Hub,** e esse custo vai para a conta (ver abaixo).

## Os dois desenhos

[`tests/kernel/strata_livro.py`](../tests/kernel/strata_livro.py) grava cada nível como um GGUF com as matrizes das camadas em F16. Assim a qualidade é medida pelo `llama-perplexity` sem kernel novo. Todo ajuste minimiza Σ imatrix_j · (x_ij − q_ij)², o mesmo critério do `llama-quantize`.

| desenho | o que guarda | bits/peso na RAM (nível de b bits) |
|---|---|---|
| **linha** (Any-Precision puro) | um livro de 2^b valores F16 por linha da matriz, k-means ponderado 1D; cada nível divide cada célula em duas com 2-médias dentro dela | b + 16·2^b / colunas (8B, 4096 colunas: 4,06 / 5,13 / 6,25) |
| **bloco** (híbrido estilo IQ4_XS) | uma escala por bloco de 32 pesos (6 bits com sinal × um F16 por 256, a conta do IQ4_XS) e um livro por tensor, ajustado com escalas e livro em rodadas alternadas, partindo da tabela do IQ4_NL; os filhos dividem cada valor do livro, por tensor ou por linha | b + 0,25 (+ tabela por linha, se os filhos forem por linha) |

Para separar o preço de aninhar do resto, `--independente` ajusta cada largura do zero, sem aninhar.

Todas as matrizes das camadas usam a mesma largura. O Q4_K_M dá 6 bits a algumas matrizes sensíveis, então essa escolha é conservadora: se o livro empatar sem a mistura, a mistura só ajuda.

## Régua de decisão (definida antes de rodar)

Eixo x: bits por peso nas matrizes das camadas, contando escalas e tabelas (para os k-quants, os bytes reais do GGUF). Eixo y: KL ×1000 contra o BF16. Os dois lados ficam no caminho F16, porque o livro não tem kernel e o F16 não arredonda as ativações (seção 28). A curva é a dos k-quants com imatrix (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0), interpolada pelo log da KL. O IQ4_XS com imatrix entra como ponto à parte: ele já é "escala por bloco + tabela de 16 valores" e está no Hub.

**No 1B (container), só para descartar:**

- Descarta se o melhor desenho, no mesmo tamanho, gastar mais de +0,15 bit/peso que a curva em qualquer nível.
- Descarta também se a soma dos três níveis não ficar abaixo da dos tipos D medidos do mesmo jeito.
- Passar no 1B não prova nada: o 1B já prometeu mais do que o 8B entregou quatro vezes.

**No 8B (Colab A100), quem decide:**

| resultado | condição | o que acontece |
|---|---|---|
| **sucesso** | os níveis 0, 1 e 2 a no máximo +0,05 bit/peso da fronteira (k-quants e IQ4_XS) | o preço de aninhar some; vale abrir o kernel |
| **ganho forte** | sucesso, e além disso algum nível ≥ 0,10 bit/peso abaixo da fronteira | idem, e vira o caminho principal |
| **zona cinza** | algum nível entre +0,05 e +0,10 | registra e não abre kernel; os tipos D continuam |
| **fracasso** | algum nível acima de +0,10, ou soma ≥ 0,34 (a dos tipos D) | descarta |

**Ajuste feito depois do 1B e antes do 8B,** e só para deixar a régua mais exigente:

- **O que mudou:** a fronteira ganhou pontos onde a curva tinha buracos: o Q4_K_S, o Q5_K_S e um Q6_K com as matrizes de atenção e a `ffn_down` em Q8_0 (7,24 bits/peso no 1B). O IQ4_XS passou a fazer parte da fronteira em todos os níveis, não só como condição do nível 0.
- **Por quê:** no 1B, a interpolação pelo log da KL entre o Q3_K_M e o Q4_K_M (1 bit de distância) dava ao nível 0 uma vantagem de −0,36 bit/peso. Entre o Q6_K e o Q8_0 (2 bits), dava −0,38 ao nível 2. Contra o IQ4_XS, a vantagem do nível 0 some (−0,01). Isso era do buraco na curva, não do formato.
- **O que vale independente da curva:** a **dominância**, menos bits e menos KL ao mesmo tempo. O notebook lista as dominâncias à parte.

Por que o empate é o mínimo: com os tipos D, o nível 0 é o Q4_K_M do Hub e roda em qualquer llama.cpp. O livro abre mão disso, então só compensa se tirar todo o preço dos níveis de cima.

## O custo de o estrato 0 não ser o Q4_K_M do Hub

- **Quem já baixou o Q4_K_M do Hub não reaproveita nada.** A conta da seção 31 deixa de valer: hoje, quem tem o Q4_K_M baixa 1,00 GB para o nível 1 e 6,25 GB para o BF16 exato. Com o livro, baixa o estrato 0 de novo (~4,3 GB no 8B).
- **O nível 0 não roda no llama.cpp sem patch.** Hoje roda. Todo nível precisaria de tipo novo no ggml, inclusive o mais baixo, que é o que roda nas máquinas pequenas.
- **"Um arquivo no lugar de BF16 + Q4 + Q5 + Q6" continua valendo** se o fecho até o BF16 custar o mesmo (~3,3 bits/peso), mas o arquivo deixa de poder ser montado "ao lado" de um GGUF publicado (especificação B.10).

## Kernel: o que ele compra, separado da qualidade

O kernel não muda a KL: a KL é do formato. Ele compra duas coisas:

1. **Rodar o formato.** Sem kernel, o livro só existe na emulação em F16.
2. **Velocidade.** Na geração, o gargalo é ler os pesos da memória. Então a meta é empatar com o k-quant do mesmo tamanho (régua de sempre: ≥ 0,95), não ganhar.

A RAM é a dos bits por peso da tabela acima. O kernel não tira nada disso.

Dá para medir uma parte já, sem escrever kernel: o nível 0 do desenho bloco tem o mesmo formato do IQ4_XS (escala de 6 bits por 32, códigos de 4 bits, tabela de 16 valores int8 consultada com `pshufb`), só que a tabela vem do tensor em vez de ser fixa. Então a velocidade do IQ4_XS no `llama-bench` é a velocidade do nível 0. Os níveis 1 e 2 consultam tabelas de 32 e 64 valores: duas ou quatro consultas de 16 por grupo de pesos, ou `vpermb` no AVX-512 VBMI. Esse custo a mais é o que os tipos D já mostraram na GPU (0,65× dos tipos P) e precisa de micro-benchmark antes de qualquer promessa.

Sobre o kernel "próprio, fora do llama.cpp" como caminho para democratizar: um kernel fora do ggml perde Ollama, LM Studio e os backends Metal, Vulkan e CUDA que o llama.cpp já tem. Democratizar é qualidade por byte boa *e* rodar onde as pessoas já estão. O caminho, se a qualidade passar, é um tipo novo dentro do ggml, como os tipos P e D.

## Resultados

### Triagem no Llama-3.2-1B (contêiner)

Caminho F16, KL contra o BF16 nos 40 primeiros trechos do wikitext-2 de teste, imatrix do texto de treino ([tabela completa e JSON](../tests/kernel/resultados/unsloth__Llama-3.2-1B_livro_container.md)). Bits por peso nas matrizes das camadas; erro ponderado = Σ imatrix·(x − q)² / Σ imatrix·x².

| modelo | bits/peso | KL ×1000 | mesmo 1º token | erro ponderado |
|---|---:|---:|---:|---:|
| Q3_K_M | 3,842 | 127,89 ± 2,03 | 82,5% | 1,851% |
| livro por linha, nível 0 | 4,099 | 174,58 ± 2,67 | 79,4% | 1,303% |
| **livro por bloco, nível 0** | 4,250 | 49,46 ± 1,08 | 89,0% | 0,472% |
| IQ4_XS | 4,272 | 47,68 ± 0,73 | 89,0% | 0,507% |
| Q4_K_S | 4,539 | 43,05 | | 0,428% |
| Q4_K_M | 4,802 | 38,76 ± 0,63 | 90,0% | 0,420% |
| livro por linha, nível 1 | 5,198 | 43,33 ± 0,73 | 89,5% | 0,324% |
| **livro por bloco, nível 1** | 5,250 | 17,36 ± 0,34 | 93,0% | 0,157% |
| Q5_K_S | 5,500 | 11,53 | | 0,110% |
| Q5_K_M | 5,656 | 11,79 ± 0,22 | 94,4% | 0,109% |
| **livro por bloco, nível 2** | 6,250 | 4,98 ± 0,08 | 96,0% | 0,042% |
| livro por linha, nível 2 | 6,397 | 10,89 ± 0,25 | 94,7% | 0,074% |
| Q6_K | 6,562 | 5,37 ± 0,12 | 95,9% | 0,028% |
| Q6_K com atenção e `ffn_down` em Q8_0 | 7,431 | 2,17 | 97,2% | 0,018% |
| Q8_0 | 8,500 | 0,62 ± 0,02 | 98,4% | 0,003% |

Bits a mais que a fronteira no mesmo KL (abaixo de 0 = o livro gasta menos):

| candidato | nível 0 | nível 1 | nível 2 | soma |
|---|---:|---:|---:|---:|
| curva só com Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 (a do plano original) | −0,356 | −0,128 | −0,380 | |
| **livro por bloco, fronteira da régua** | **−0,006** | **−0,014** | **−0,384** | **−0,40** |
| livro por linha, fronteira da régua | +0,393 | +0,676 | +0,817 | +1,89 |
| tipos D por tensor (seção 27, 5,80 e 6,80 bits/peso), fronteira da régua | 0 | +0,373 | −0,021 | +0,35 |
| livro por bloco **sem aninhar** (`--independente`, cada largura ajustada do zero; não é STRATA) | −0,006 | −0,200 | −0,585 | |

O que isso diz:

- **O livro por linha (Any-Precision puro) está descartado.** O nível 0 fica pior que o Q3_K_M gastando mais bits. Um livro por linha sem escala por bloco não acompanha a variação de tamanho dos pesos ao longo da linha: o erro ponderado é 2 a 3 vezes o do livro por bloco em todos os níveis.
- **O livro por bloco passou no filtro do 1B.** Nenhum nível passa de +0,15 bit/peso, e a soma (−0,40) fica abaixo da dos tipos D (+0,35).
- **Os pontos novos da curva mudaram a conta, como se temia.** O nível 1 parecia 0,13 bit/peso à frente; com o Q5_K_S na fronteira, virou empate (−0,01). O nível 0 já era empate com o IQ4_XS. Os −0,38 do nível 2 ainda vêm de uma interpolação entre o Q6_K e o Q6_K com Q8_0 (0,87 bit de distância), então podem encolher do mesmo jeito.
- **O que não depende de interpolação:** o nível 2 domina o Q6_K (6,25 contra 6,56 bits/peso, KL 4,98 contra 5,37). Só isso já é pelo menos 0,31 bit/peso a menos no mesmo KL.
- **O preço de aninhar continua lá, ~0,19 bit/peso por nível.** Ajustado do zero, o mesmo formato de 5,25 bits/peso dá KL 12,58 (contra 17,36 aninhado) e fica 0,20 bit/peso à frente da fronteira. Com 6,25 bits/peso, dá 4,04 contra 4,98. Ou seja, o livro melhorou o formato em ~0,2 bit/peso, e o aninhamento devolveu quase tudo no nível 1. É da mesma ordem dos +0,11 a +0,23 que as vertentes anteriores pagavam contra os k-quants no 8B; o livro só começou de um ponto melhor.
- **Filhos por linha não resolvem no 1B:** a tabela por linha custa 16·2^b/colunas, 0,25 bit/peso no nível 1 com 2048 colunas, mais que o preço que ela tentaria tirar. No 8B (4096 colunas) custa 0,125 e 0,25; o notebook tem a variante `bloco_linha`, desligada por padrão.
- **Em resumo, no 1B:** empate nos níveis 0 e 1, ganho no nível 2. Pela régua do 8B isso seria "ganho forte". Mas é o padrão que já falhou no 8B (seções 26 a 30: o 1B prometia no nível 2 e o 8B não entregou). Quem decide é o notebook.

### Kernel no contêiner (CPU, AVX2)

`llama-bench` do llama.cpp sem patch, Llama-3.2-1B, tok/s:

| | geração 1 thread | geração 4 threads | prompt 1 thread | prompt 4 threads |
|---|---:|---:|---:|---:|
| Q4_K_M | 6,74 | 25,54 | 76,6 | 327 |
| IQ4_XS | 6,95 | 24,51 | 108,9 | 370 |
| IQ4_XS / Q4_K_M | 1,03 | 0,96 | 1,42 | 1,13 |

O nível 0 do livro por bloco tem o formato do IQ4_XS (só a tabela de 16 valores muda), então a velocidade dele é essa: a tabela de 16 sai de graça.

Micro-benchmark do produto escalar ([`tests/kernel/bench_livro.c`](../tests/kernel/bench_livro.c), conferido contra uma referência escalar), 16384 × 4096 pesos, ns por peso, 1 thread, melhor de várias rodadas (o contêiner é ruidoso, ±20%):

| | bits/peso | ns/peso | em relação ao nível 0 |
|---|---:|---:|---:|
| Q4_K (ggml) | 4,50 | 0,039 | |
| IQ4_XS (ggml) | 4,25 | 0,046–0,053 | |
| Q5_K (ggml) | 5,50 | 0,044–0,052 | |
| Q6_K (ggml) | 6,56 | 0,033–0,045 | |
| livro nível 0 (tabela de 16, `pshufb`) | 4,25 | 0,071–0,080 | 1 |
| livro nível 1 (tabela de 32: 2 `pshufb` + `blendv`) | 5,25 | 0,087–0,101 | ~1,2–1,3 |
| livro nível 2 (tabela de 64: 4 `pshufb` + 3 `blendv`) | 6,25 | 0,115–0,134 | ~1,5–1,7 |

- **Meu nível 0 ainda é ~1,5× mais lento que o IQ4_XS do ggml,** com o mesmo formato. Falta otimização. Então os números absolutos não valem; vale a razão entre os níveis.
- **Tabelas de 32 e 64 custam ~1,2× e ~1,5× no cálculo (com o ruído, até 1,3× e 1,7×).** Na geração, o gargalo costuma ser ler a memória, e o custo a mais se esconde em parte. Na GPU, os tipos D (que também consultam tabela) ficaram em 0,65× dos tipos P (seção 30). Essa é a estimativa honesta até existir kernel de verdade.
- **O processador do contêiner tem AVX-512 VBMI** (`vpermb` consulta 64 bytes de uma vez). Lá a tabela de 64 custaria uma instrução, mas a maioria das máquinas de consumo não tem.

### Llama-3.1-8B (Colab A100)

Aguardando o [notebook](../tests/kernel/strata_8b_livro_colab.ipynb).
