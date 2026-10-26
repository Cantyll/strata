# Roteiro: vídeo comercial do STRATA (9:16, 3 min 51 s)

Apresentação de produto com tudo o que foi construído e medido até o PR #11, guiada por um personagem, o **Bit**: um robozinho feito dos cinco estratos do logo (a base dourada é o Q4_K_M), com visor, olhos que piscam e olham para o que ele aponta, e uma boca que segue o volume da narração. Ele fica fora do cubo das transições, então atravessa as cenas voando, e anda pelo corredor da direita até o elemento de que a narração fala, apontando com um feixe pontilhado.

- Narração: ElevenLabs, voz Henrique - Clear and Knowledgeable, `eleven_v4`, um parágrafo por pedido, uma tomada cada (3.321 créditos no total). Texto falado em [`fonte/narracao_tts.txt`](fonte/narracao_tts.txt).
- Padrão visual: `video/README.md` (PR #11). Duração acima dos 180 s do padrão a pedido do Cantyll para este vídeo.
- Regra do final: não há cena de veredito. O vídeo termina no próximo passo e na despedida do Bit; a conclusão só é gravada depois do merge.

Legendas exibidas (como na tela) por cena, com a fonte de cada número (as seções 31.x de `docs/05` chegam com o PR #11):

## 1. Apresentação: o Bit (0.0–9.8 s)

Oi! Eu sou o Bit, e vou ser o seu guia pelo STRATA.
Em alguns minutos, você vai ver o que construímos,
o que medimos, e onde ainda perdemos.
Vem comigo.

Números: —.

## 2. O problema (10.6–29.3 s)

Modelos de IA são enormes, e a internet guarda cada um várias vezes.
O Llama de 8 bilhões de parâmetros aparece no Hugging Face
como o original e como três versões comprimidas:
33 GB no total.
Num único repositório, achamos 29 versões do mesmo modelo.

Números: docs/01 (Hub: 29 versões, 191 GB); docs/05 sec. 31 (8B: 4 arquivos, 33,31 GB).

## 3. A ideia (30.1–48.0 s)

A ideia do STRATA é guardar só a surpresa:
só o que ainda não dá para prever.
Os bits de cada peso viram camadas,
do mais importante ao menos importante.
Ler o começo já dá um modelo.
Ler tudo devolve o original, bit a bit, com a mesma impressão digital.

Números: README; docs/05 sec. 1 e 12.

## 4. Passo 1 · sem perdas (48.8–67.4 s)

Primeiro passo: o arquivo sem perdas.
O expoente de cada peso ocupa 8 bits,
mas carrega só 2,6 de informação.
Um codificador chamado rANS guarda só isso.
No Llama de 1 bilhão, o arquivo ficou com 66% do tamanho original,
e voltou idêntico.

Números: docs/05 sec. 12 (expoente 2,607 b; rANS 2,6075; 10,594 b/peso = 66,2%; SHA-256 idêntico).

## 5. Sendo honesto: o corte perdeu (68.2–86.2 s)

Depois, a pergunta difícil:
um pedaço do arquivo compete com os formatos comprimidos que todo mundo usa,
os k-quants do llama.cpp?
A resposta foi não.
O Q4_K_M errou 3 vezes menos que o nosso corte do mesmo tamanho.
A diferença era estrutural.

Números: docs/05 sec. 13 (Q4_K_M 4,80 b +4,07% vs E1 4,59 b +13,08%; 3,6 b de sinal+expoente).

## 6. Formato 0.3: a nova base é o Q4_K_M (87.0–105.8 s)

Então mudamos a base.
No formato 0.3, a camada 0 é o próprio Q4_K_M, byte a byte.
Cada nível acima custa 1 bit por peso
e divide o degrau da régua ao meio.
A última camada, o fecho, completa o original exato.
Tudo em cerca de 11 bits por peso.

Números: docs/05 sec. 15, 16 e 21 (11,12 b/peso 1B; 11,13 no 8B).

## 7. Segurança (106.6–126.8 s)

Um formato novo precisa ser seguro.
Atacamos o leitor com 30 mil arquivos adulterados.
No começo, quase 1 em cada 5 passava sem erro, com pesos errados.
Reescrevemos o leitor em Rust, com uma impressão digital para cada pedaço.
Hoje, nenhum passa em silêncio:
mudou um byte, o arquivo é recusado.

Números: docs/05 sec. 18 (30.000 casos; 18,9% silenciosos -> 0 com hashes).

## 8. Vários arquivos e leitura em fluxo (127.6–147.3 s)

Modelos grandes chegam em vários arquivos,
e cada um volta com a mesma impressão digital do Hugging Face.
E não é preciso esperar o download acabar:
o STRATA monta o modelo enquanto baixa.
No Llama de 8 bilhões, a 100 megabits por segundo,
o original fica pronto em 8 minutos e meio, contra 21 baixando direto.

Números: docs/05 sec. 19 e 31.3 (4 shards com o SHA do HF; 8B a 100 Mbit/s: 518 s vs 1.286 s).

## 9. Dentro do llama.cpp (148.1–168.5 s)

E os níveis do meio, rodam?
Criamos tipos novos dentro do llama.cpp, no processador e na placa de vídeo.
Na placa de vídeo, o nível 1 gera texto na velocidade do Q5_K_M.
Mas, na mesma memória, a qualidade ainda fica atrás dos k-quants com imatrix:
de 1 a 2 décimos de bit por peso.

Números: docs/05 sec. 24 e 30 (A100: nível 1 144,0 vs Q5_K_M 144,6 tok/s; tipos D +0,23 / +0,11 b/peso; D a 0,65x na GPU).

## 10. O ganho real: armazenamento (169.3–192.2 s)

Onde o STRATA ganha de verdade é no armazenamento.
O Q4_K_M do Hub fica intacto, como camada zero.
No Llama de 8 bilhões, 4 arquivos com 33 GB viram um conjunto de 11:
66% a menos.
No de 70 bilhões, de 291 para 98 GB,
com os 30 arquivos de volta, bit a bit.

Números: docs/05 sec. 31 e 31.1 (8B 33,31 -> 11,18 GB; 70B 291,47 -> 98,27 GB, 30/30 shards).

## 11. Já tem o Q4_K_M? (193.0–200.2 s)

E quem já tem o Q4_K_M?
Baixa só 6 GB de camadas para ter o original, em vez de 16.

Números: docs/05 sec. 31 (6,25 GB vs 16,06 GB, -61%).

## 12. Os limites (201.0–219.7 s)

Agora, os limites, sem enfeite.
Os níveis do meio ainda perdem para os k-quants
e precisam de um llama.cpp modificado.
Em rede rápida, acima de uns 670 megabits,
baixar o original direto ainda é mais rápido.
E testamos só a família Llama, com a rede simulada.

Números: docs/05 sec. 28, 30, 31, 31.3 (empate ~670 Mbit/s; rede simulada).

## 13. Próximo passo e despedida (220.5–228.7 s)

O próximo passo é publicar as camadas no Hub de verdade
e medir com outros modelos.
Eu sou o Bit. Até a próxima etapa do STRATA.

Números: docs/05 sec. 31.3 (camadas ainda fora do Hub).
