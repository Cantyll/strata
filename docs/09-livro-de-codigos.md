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

(preenchido depois das medidas)
