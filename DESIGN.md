---
name: STRATA
description: Painel de azulejos de estação; cobalto sobre branco-leite vidrado, pintado no navegador.
colors:
  glaze: "#EEE7D7"
  grout: "#D6CDB9"
  glaze-shade: "#786850"
  ink: "#0E2463"
  ink-soft: "#33508F"
  paint: "#183A97"
  paint-deep: "#10286E"
  wash: "#5873BE"
  wash-pale: "#CFD6E8"
  mustard: "#C99A2E"
  risco: "#6E7FA6"
  plaque-fg: "#F7F3EA"
  focus: "#0E2463"
  bad: "#9E2A1B"
  glaze-meia-luz: "#B5AE9E"
  grout-meia-luz: "#9C9483"
  glaze-shade-meia-luz: "#463C2D"
  ink-meia-luz: "#0A1A4C"
  ink-soft-meia-luz: "#1F3672"
  paint-meia-luz: "#143080"
  paint-deep-meia-luz: "#0C2060"
  wash-meia-luz: "#3F5BA8"
  wash-pale-meia-luz: "#A8B2CB"
  mustard-meia-luz: "#A57C1D"
  risco-meia-luz: "#4C5C88"
  plaque-fg-meia-luz: "#EDE7DA"
  focus-meia-luz: "#0A1A4C"
  bad-meia-luz: "#7A1E12"
typography:
  display:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: "clamp(1.35rem, calc(var(--tile) * .54), 3rem)"
    fontWeight: 400
    lineHeight: 1.08
    letterSpacing: "0.025em"
  headline:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: "clamp(1.15rem, calc(var(--tile) * .43), 2.05rem)"
    fontWeight: 400
    lineHeight: 1.08
    letterSpacing: "0.025em"
  marca:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: "clamp(.9rem, calc(var(--tile) * .34), 1.4rem)"
    fontWeight: 400
    letterSpacing: "0.34em"
  title:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: "1.2rem"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0.16em"
  figure:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: "1.3rem"
    fontWeight: 400
    letterSpacing: "0.04em"
    fontFeature: "\"tnum\", \"lnum\""
  body:
    fontFamily: "Castoro, Georgia, Times New Roman, serif"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.5
  body-small:
    fontFamily: "Castoro, Georgia, Times New Roman, serif"
    fontSize: ".9rem"
    fontWeight: 400
    lineHeight: 1.5
  button:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: "clamp(.8rem, calc(var(--tile) * .33), 1.05rem)"
    fontWeight: 400
    letterSpacing: "0.08em"
  label:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: ".8rem"
    fontWeight: 400
    letterSpacing: "0.1em"
  label-small:
    fontFamily: "Castoro Titling, Castoro, Georgia, Times New Roman, serif"
    fontSize: ".72rem"
    fontWeight: 400
    letterSpacing: "0.07em"
rounded:
  none: "0px"
  hair: "1px"
spacing:
  tile: "var(--tile)"
  tile-min: "30px"
  tile-max: "88px"
  gutter: "16px"
  rejunte: "2px"
  fiada-inset: "calc(var(--tile) * .62)"
  moldura: "calc(var(--tile) * .34)"
components:
  placa:
    backgroundColor: "{colors.paint}"
    textColor: "{colors.plaque-fg}"
    typography: "{typography.button}"
    rounded: "{rounded.none}"
    padding: ".65rem clamp(.6rem, calc(var(--tile) * .3), .9rem) .6rem"
    height: "max(3rem, calc(var(--tile) * 1.1))"
    width: "100%"
  placa-hover:
    backgroundColor: "{colors.paint-deep}"
    textColor: "{colors.plaque-fg}"
  placa-clara:
    backgroundColor: "{colors.glaze}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.none}"
    width: "100%"
  placa-clara-hover:
    textColor: "{colors.paint}"
  letreiro:
    backgroundColor: "{colors.paint}"
    textColor: "{colors.plaque-fg}"
    rounded: "{rounded.none}"
    padding: ".5rem .95rem .45rem"
  cartela-titulo:
    textColor: "{colors.paint}"
    typography: "{typography.display}"
    padding: "calc(var(--tile) * .44) calc(var(--tile) * .5) calc(var(--tile) * .38)"
  grade-hash-azulejo:
    backgroundColor: "{colors.glaze}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
  fileira-azulejo:
    backgroundColor: "{colors.paint}"
    rounded: "{rounded.none}"
    size: "11px"
  fileira-azulejo-vazio:
    backgroundColor: "{colors.glaze}"
    size: "11px"
  ficha-herda:
    textColor: "{colors.ink-soft}"
    typography: "{typography.label-small}"
    rounded: "{rounded.hair}"
    padding: ".26rem .5rem .2rem"
  ficha-leve:
    backgroundColor: "{colors.wash-pale}"
    textColor: "{colors.ink}"
    typography: "{typography.label-small}"
    rounded: "{rounded.hair}"
    padding: ".26rem .5rem .2rem"
  ficha-total:
    backgroundColor: "{colors.paint}"
    textColor: "{colors.plaque-fg}"
    typography: "{typography.label-small}"
    rounded: "{rounded.hair}"
    padding: ".26rem .5rem .2rem"
  summary:
    textColor: "{colors.ink}"
    typography: "{typography.label}"
---

# Design System: STRATA

## Overview

**Creative North Star: "O Painel de Estação"**

A página inteira é uma parede de azulejos de estação ferroviária portuguesa: branco-leite vidrado, cortado por rejunte em azulejos inteiros, e pintado à mão em cobalto. Cada bloco de pesos do modelo é um azulejo; cada estrato do arquivo é uma demão de cobalto. Azulejo em branco quer dizer "ainda não pintado"; cobalto profundo quer dizer "ativo" ou "conferido". O sistema não imita uma interface sobre um fundo decorativo: os ornamentos (frisos, molduras, cantoneiras, cartelas, placas) são a interface, e são pintados pelo próprio navegador, no tamanho real do azulejo e nas cores do tema.

A densidade é de painel, não de dashboard: uma coluna de painéis emoldurados no celular, painéis em duas colunas a partir de 960px, frisos de um azulejo separando um painel do outro. Títulos moram em cartelas de volutas, em maiúsculas romanas; o corpo é um serifado de livro, legível, em frases de leigo. A meia-luz (tema escuro) é o mesmo vidrado com a luz baixa, não um fundo preto: o cobalto continua sobre o branco-leite.

Recusa confirmada pela direção: a demo de IA padrão (fundo escuro, gradiente, cards de KPI).

**Key Characteristics:**
- Uma parede: tudo encaixa em fiadas de azulejos inteiros, e o rejunte passa por cima do que foi pintado.
- Dois pigmentos: cobalto sobre branco-leite; mostarda só como filete dentro dos ornamentos pintados.
- Ornamentos rasterizados em tempo de execução pelo `strata-pincel.js`; nenhuma imagem externa é enviada.
- Pinturas compostas em multiplicar, para que o vidrado e o craquelê apareçam por baixo.
- Castoro Titling em maiúsculas para tudo o que é inscrição; Castoro para tudo o que é leitura.
- Movimento de forno: a demão de cobalto lava os azulejos em diagonal.

## Colors

Uma paleta de vidrado e pigmento: um branco-leite quente, um cobalto em três forças, e uma mostarda que só aparece como filete de ornamento.

### Primary
- **Cobalto de Pintura** (paint): a cor da pintura. Títulos das cartelas, a marca, o fundo das placas e do letreiro, os azulejos pintados das fileiras, o contorno da grade do SHA-256, o valor do controle de demãos e os números grandes. Também é a cor-base com que o pincel pinta todos os ornamentos e a albarrada.
- **Cobalto Profundo** (paint-deep): a segunda carga de pigmento; o pincel usa-o nas partes mais densas dos ornamentos e na placa escura de hover.
- **Aguada de Cobalto** (wash): contorno dos estados ainda não preenchidos (pastilha das etapas, azulejo vazio das fileiras, fichas de herança).
- **Aguada Clara** (wash-pale): o cobalto diluído quase em branco. Fundo da ficha de ajuste leve, a seleção de texto e o pico do pulso da etapa atual.

### Secondary
- **Mostarda de Filete** (mustard): só existe dentro dos ornamentos pintados (filetes da moldura, do friso, das cartelas, das placas, botões das cantoneiras) e no contorno da grade do SHA-256 quando a conferência termina. Nunca em texto, nunca como fundo.

### Neutral
- **Branco-Leite Vidrado** (glaze): o vidrado de cada azulejo, o fundo da página, o halo das letras e o "azulejo em branco" (não pintado).
- **Rejunte** (grout): as linhas de rejunte que cortam a parede, o fundo entre os azulejos da grade do SHA-256 e os filetes horizontais de tabela, lista e `details`.
- **Sombra do Vidrado** (glaze-shade): só entra no pincel, para a almofada e o craquelê do azulejo de fundo.
- **Tinta** (ink): o texto de leitura, títulos de colunas não-cobalto e o anel de foco (focus tem o mesmo valor).
- **Tinta Suave** (ink-soft): notas, legendas, rótulos, etapas ainda não feitas.
- **Risco** (risco): o estresido, o contorno pontilhado de antes da pintura. Pinta a albarrada em branco, o friso-risco do trilho e a borda tracejada das mensagens de espera.
- **Vidrado de Placa** (plaque-fg): o texto claro sobre as placas, o letreiro e a ficha de total.
- **Óxido** (bad): só para falha de conferência (azulejo que não bateu e selo de erro).

Cada token tem uma versão de meia-luz (sufixo `-meia-luz` no frontmatter), ativada por `prefers-color-scheme: dark` ou por `data-theme="dark"` na raiz; `data-theme="light"` força o claro. O pincel lê as cores do tema e repinta todos os ornamentos quando o tema muda.

### Named Rules
**The Dois Pigmentos Rule.** Cobalto sobre branco-leite, só. A mostarda é filete de ornamento pintado e o sinal de "conferido" na grade do SHA-256; nunca vira texto, fundo ou botão.

**The Branco É Não-Pintado Rule.** Azulejo em branco com contorno de aguada quer dizer "ainda não"; azulejo cheio de cobalto quer dizer "lido", "ativo" ou "conferido"; azulejo hachurado em diagonal quer dizer "estimado, não medido". Esse código vale para fileiras, pastilhas e trilho.

**The Meia-Luz Rule.** O tema escuro baixa a luz do mesmo vidrado; não inverte a parede para fundo preto.

## Typography

**Display Font:** Castoro Titling (com Castoro, Georgia, Times New Roman, serif)
**Body Font:** Castoro (com Georgia, Times New Roman, serif)

**Character:** Castoro Titling é a letra de inscrição, maiúscula romana como a de uma legenda pintada em painel; Castoro é o serifado de livro, quente e legível em frases inteiras. As duas vêm do Google Fonts.

### Hierarchy
- **Display** (400, clamp ligado ao azulejo, 1.08): o título da abertura dentro da cartela, sempre em maiúsculas, com um traço de 0.028em que engrossa a letra como pincel. A partir de 960px o fator cai para `tile * .5`.
- **Headline** (400, clamp ligado ao azulejo, 1.08): títulos de seção dentro das cartelas; `tile * .36` a partir de 960px.
- **Marca** (400, espaçamento 0.34em): STRATA, em cobalto, na pequena cartela do friso de topo.
- **Title** (400, 1.2rem, 0.16em): a palavra do selo (IDÊNTICO).
- **Figure** (400, 1.3rem, números tabulares): os números grandes do controle de demãos.
- **Body** (400, 1.0625rem, 1.5): todo texto de leitura; medida máxima de 36rem (30rem nas frases de apoio da abertura), `text-wrap: pretty`.
- **Body Small** (400, .9rem): notas, legendas, tabela.
- **Button** (400, clamp ligado ao azulejo, 0.08em, maiúsculas): texto das placas.
- **Label** (400, .8rem, 0.1em, maiúsculas): rótulos de controle e o `summary`; **Label Small** (.72rem, 0.07em) para etapas, cabeçalhos de tabela e fichas.

### Named Rules
**The Inscrição e Leitura Rule.** Castoro Titling em maiúsculas é para o que se inscreve (títulos, marca, placas, rótulos de dado); Castoro em caixa normal é para o que se lê. Uma frase inteira nunca vai em Castoro Titling.

**The Número Medido Rule.** Todo número medido na página usa algarismos tabulares e alinhados (`tabular-nums lining-nums`) e formatação pt-BR.

## Layout

A parede é uma grade de azulejos quadrados. O lado do azulejo (`--tile`) sai da largura do contêiner: 8 colunas abaixo de 960px e 14 colunas a partir de 960px (consulta de contêiner sobre o `html`, não media query). O tamanho é limitado pela altura para que a abertura caiba na primeira tela (`(100svh - 24px) / 15` no celular, com piso de 90% da largura ou 40px; `/ 9` na tela larga), fica entre 30px e 88px, e é arredondado para baixo até px inteiro onde `round()` existe. A parede (`.parede`) tem exatamente `tile × colunas` de largura, centrada, com 16px de margem lateral na página.

Toda fiada marcada com `data-snap` tem a altura arredondada pelo JavaScript para um número inteiro de azulejos, a cada conversão, troca de demão, carregamento de fontes e redimensionamento. Os paddings internos são frações do azulejo (`.62`, `.45`, `.3`), não uma escala fixa; a última fiada de cada painel deixa `1.2` azulejo livre para a cantoneira.

No celular, as colunas se desfazem e as fiadas descem na ordem da leitura (texto, pintura, ação). A partir de 960px cada painel vira duas colunas de 6 + 8 azulejos; o painel de demãos usa 1 + 8 + 4 + 1 com o título centrado em 10 azulejos; painéis únicos centram 8 azulejos. As pinturas têm 8×6 azulejos; a lupa, 3 azulejos de altura; frisos e trilho, 1 azulejo.

### Named Rules
**The Azulejo Inteiro Rule.** Nada termina no meio de um azulejo: largura da parede em colunas inteiras, fiadas encaixadas em azulejos inteiros, `--tile` em px inteiro. Um componente novo entra numa fiada `data-snap` ou mede um número inteiro de azulejos.

## Elevation & Depth

Não há `box-shadow` no sistema. A profundidade é a de uma parede real, em três camadas: o vidrado com almofada e craquelê pintados (fundo `--azulejos`), as pinturas e ornamentos compostos em `mix-blend-mode: multiply` por cima dele, e o rejunte (`body::after`, z-index 4, opacidade .72) passando por cima de tudo o que foi pintado. O que se lê e os azulejos de dados que se contam ficam acima do rejunte, em z-index 6; letras sobre a parede ganham um halo do próprio vidrado que interrompe o rejunte atrás delas. A moldura fica em z-index 2, sobre o conteúdo pintado e sob o texto.

### Shadow Vocabulary
- **Halo do Vidrado** (`text-shadow: 0 0 1px var(--glaze), 0 0 3px var(--glaze), 0 0 5px var(--glaze)`): em todo texto de leitura, rótulo e célula de tabela que fica sobre a parede. Não é sombra de elevação; é o vidrado limpo em volta da letra.

### Named Rules
**The Rejunte Por Cima Rule.** O rejunte cruza toda pintura, ornamento e placa, como numa parede de verdade. Só o que precisa ser lido ou contado sobe acima dele (z-index 6), e o texto leva o halo do vidrado.

**The Multiplicar Rule.** Pinturas (canvas) e ornamentos (frisos, moldura, trilho) entram em `multiply`, para que o vidrado e o craquelê apareçam através do cobalto.

## Shapes

Tudo é quadrado: o azulejo, a pastilha das etapas, os azulejos das fileiras e da grade do SHA-256, as placas e o seletor (raio 0). As únicas curvas vêm do pincel: volutas, florão, folhas das cantoneiras, as quinas chanfradas da cartela. As bordas emolduradas são `border-image` em nove fatias, com a fatia (`--moldura-corte`, `--cartela-corte`, `--placa-corte`) informada pelo pincel; a moldura repete (`round`), cartelas e placas esticam (`stretch`), e a placa usa `fill` para o miolo. Contornos de estado são `outline` com deslocamento negativo, para não mudar a medida do azulejo. As fichas têm um raio de 1px, o único arredondamento do sistema.

## Components

### Placa (botão)
Uma placa de cartela em cobalto, pintada.
- **Forma:** largura cheia, altura mínima `max(3rem, 1.1 azulejo)`, borda em nove fatias com miolo (`--img-placa ... fill`). Sem a pintura, cai para um retângulo cobalto chapado.
- **Primária:** texto Vidrado de Placa em Castoro Titling maiúsculo; ícone de seta em SVG de traço.
- **Hover / Active / Foco:** hover troca para a placa escura pintada; active desce 1px (.12s); foco é o anel global de 3px em Tinta com 3px de afastamento. Desabilitada: opacidade .78 e cursor de progresso.
- **Convite:** antes da primeira conversão, a seta empurra 5px para a direita em ciclo de 1.5s.
- **Placa clara:** placa de vidrado com filetes, texto em Tinta; hover passa o texto para Cobalto. Para a ação secundária do fecho.

### Friso e Marca
Faixa de 1 azulejo de altura com o motivo de friso pintado e repetido por azulejo, em multiplicar. Separa um painel do outro e abre e fecha a parede. O friso de topo leva a marca STRATA numa pequena cartela pintada sobre fundo de vidrado.

### Painel, Moldura e Cantos
Cada seção é um painel emoldurado: a moldura pintada (nove fatias, `round`, espessura de .34 azulejo, multiplicar) cobre o painel inteiro, e quatro cantoneiras pintadas de .84 azulejo ficam a .28 azulejo das quinas, espelhadas por `transform`.

### Cartela
O suporte dos títulos: cartela de nove fatias esticada, com um florão pintado sobre a borda superior (só na abertura) e uma voluta de cada lado, espelhada. Título em Cobalto, centrado.

### Letreiro
Placa pequena, cobalto, ancorada no canto da pintura da abertura durante a conversão: "Demão N de M" e os kB lidos. Entra subindo 3px em .3s a cada demão.

### Grade do SHA-256
O momento de prova. 64 azulejos em 8×8, separados por 2px de rejunte, com contorno de 2px em Cobalto que vira Mostarda quando a conferência termina. Cada azulejo traz um caractere hexadecimal em Castoro Titling e é pintado no tom do seu valor (`color-mix` de cobalto com vidrado, em dois blocos de tom para manter contraste). Azulejos conferidos batem com uma pequena escala (.42s); um azulejo que não bateu leva contorno de Óxido.

### Selo
O veredito: cartela de nove fatias com um visto em SVG e a palavra IDÊNTICO em Castoro Titling Cobalto, mais "64 de 64 azulejos batem". Entra como um carimbo (escala 1.14 para 1, .32s). Em erro, palavra e visto em Óxido.

### Fileiras de Azulejos
Medidas de tamanho contadas em azulejos de 11px (ou em grade responsiva de `--n` colunas na versão grande), com 2px de rejunte. Cobalto = ocupado; **vazio** = vidrado com contorno de aguada; **estimado** = hachura diagonal de cobalto sobre vidrado com contorno cobalto, para tudo o que é estimativa e não medição.

### Trilho de Demãos
O controle de assinatura. Oito azulejos de friso em fila: o friso-risco pontilhado para demãos não lidas e o friso pintado para as lidas. Por cima, um `input range` transparente do tamanho do trilho, cujo polegar é o seletor pintado, um azulejo inteiro. Foco: anel de 3px. Mexer no trilho repinta a pintura de demãos, a lupa, os números e o friso juntos.

### Lupa
Recorte ampliado 3× da faixa de bolinhas do vaso, 3 azulejos de altura, sem suavização (os pixels da demão ficam visíveis), em multiplicar. A pintura principal marca com um retângulo tracejado em Tinta de onde a lupa olha.

### Etapas
Lista de passos da conversão (ler, separar demãos, comprimir, remontar, conferir) em Label Small maiúsculo, cada um com uma pastilha quadrada de 10px: contorno de aguada quando pendente, cheia de Cobalto quando feita, pulsando até a Aguada Clara (.9s) na etapa atual.

### Fichas
Etiquetas de tipo de parte na família: **herda** (contorno de aguada, texto suave), **leve** (fundo de aguada clara), **total** (cobalto cheio, texto claro). Sem halo.

### Detalhes (details/summary)
Detalhe técnico só para quem pede: filete de rejunte em cima, `summary` em Label maiúsculo com um chevron SVG que gira 90° ao abrir (.2s). Dentro, tabela com números tabulares e filetes de rejunte.

### Mensagem de Espera
Antes da conversão, as áreas de resultado mostram uma caixa com borda tracejada em Risco, ícone de seta em SVG e uma frase dizendo o que fazer. É o estresido de uma pintura ainda por fazer.

### Rasters e Proveniência
Nenhuma imagem externa é enviada. Todo raster da página é gerado no aparelho, no carregamento ou na conversão:

| Raster | Proveniência |
|---|---|
| `--azulejos` (vidrado, almofada, craquelê; bloco de 4×4 azulejos, JPEG) | Pintado em tempo de execução por `strata-pincel.js` |
| `--img-friso`, `--img-friso-risco` | Pintado em tempo de execução por `strata-pincel.js` (modo pinta e modo risco) |
| `--img-canto`, `--img-florao`, `--img-voluta` | Pintado em tempo de execução por `strata-pincel.js` |
| `--img-moldura`, `--img-cartela` (+ `--moldura-corte`, `--cartela-corte`) | Pintado em tempo de execução por `strata-pincel.js` |
| `--img-placa`, `--img-placa-escura`, `--img-placa-clara` (+ `--placa-corte`) | Pintado em tempo de execução por `strata-pincel.js` |
| `--img-seletor` | Pintado em tempo de execução por `strata-pincel.js` |
| Albarrada em risco (painel em branco, lupa antes da conversão) | Desenhada em código na página (`drawScene`, modo risco) |
| Albarrada pintada, demão a demão (abertura, painel de demãos, lupa) | Lida dos pesos do modelo: o desenho é embutido no modelo sintético e cada demão é reconstruída do `.strata` e colorida em cobalto |

O pincel pinta tudo no tamanho real do azulejo (até 2× de densidade de pixel) e nas cores do tema, e repinta quando o tamanho do azulejo ou o tema mudam.

### Movimento
- **Lavagem de cobalto:** cada demão entra azulejo a azulejo numa frente diagonal (coluna + linha), com borda suave de 2.2 azulejos e saída cúbica; 760ms na primeira demão, 420ms nas seguintes.
- **Carimbo** do selo (.32s), **batida** dos azulejos do SHA-256 (.42s), **pulso** da pastilha atual (.9s), **convite** da seta da placa (1.5s), **descida** da seta "Ver a prova" (1.6s).
- Curva padrão `cubic-bezier(.16, 1, .3, 1)`.
- Movimento reduzido: o CSS zera animações e transições; o JavaScript mostra a demão final sem lavagem, não toca a batida, zera as pausas e rola sem suavidade.

## Do's and Don'ts

### Do:
- **Do** medir tudo em azulejos (`--tile`) e encaixar cada fiada nova em `data-snap` para que ela termine num azulejo inteiro.
- **Do** pedir todo ornamento novo ao `strata-pincel.js`, no tamanho do azulejo e nas cores do tema, e expô-lo como propriedade `--img-*` com fallback `none`.
- **Do** compor pinturas e ornamentos em `mix-blend-mode: multiply` sobre o vidrado.
- **Do** colocar texto e azulejos de dados em z-index 6, acima do rejunte, com o halo do vidrado.
- **Do** usar azulejo em branco para "ainda não", cobalto cheio para "lido/conferido" e hachura diagonal para "estimado".
- **Do** pôr títulos em cartela, em Castoro Titling maiúsculo e Cobalto; corpo em Castoro, frases de leigo em pt-BR, sem jargão no primeiro contato.
- **Do** respeitar `prefers-reduced-motion` em CSS e no JavaScript.
- **Do** registrar a proveniência de todo raster novo: pintado pelo pincel, desenhado em código ou lido dos pesos.

### Don't:
- **Don't** usar `box-shadow` ou gradiente de CSS para imitar relevo, brilho ou pincelada; isso sai do pincel. Gradiente de CSS só traça geometria: a linha de rejunte e a hachura do estimado.
- **Don't** usar mostarda em texto, fundo ou botão.
- **Don't** escurecer a parede para preto na meia-luz; baixe a luz do mesmo vidrado.
- **Don't** arredondar cantos de azulejos, placas ou contornos de estado.
- **Don't** enviar imagens externas.
- **Don't** fazer a demo de IA padrão: fundo escuro, gradiente, cards de KPI.
