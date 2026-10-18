# Histórico

Como o STRATA saiu de um pedido e virou formato, código, protótipo e este repositório. Tudo foi feito numa única sessão de trabalho com o Claude, em 10 de outubro de 2026; o repositório foi montado na virada para o dia 11.

## O pedido

> pesquisa aprofundada sobre novo formato de armazenamento de informações que fuja dos conceitos comuns usados hoje em dia. Propósito da pesquisa, Resolver o problema de armazenamento de grandes softwares como modelos LLMS de alto calibre, o novo formato deve ser compatível com a conversão direta do formato atual de armazenamento para o novo formato. A ideia aqui é pensar fora da caixa. Um protótipo em html pode ser feito ao final seguindo os padrões de design do plugin /impeccable

Três exigências saíram daí: um formato que fuja dos conceitos atuais, conversão direta a partir do formato de hoje e um protótipo em HTML com o padrão de design do plugin Impeccable.

## Linha do tempo

### 1. Estado da arte

Levantamento de trabalhos de 2024 a 2026 em quatro frentes: compressão sem perdas de números de ponto flutuante (ZipNN, DFloat11, ECF8), precisão aninhada (Any-Precision LLM, Matryoshka Quantization, TRACE), genealogia de fine-tunes (ZipLLM + BitX, BitDelta) e representações procedurais ou estruturais (SeedLM, MXFP4, CompactifAI). Mais os números de armazenamento do Hugging Face: quase 45 PB em 2 milhões de repositórios e um único repositório com 29 quantizações do mesmo modelo, somando 191 GB.

Conclusão que guiou o resto: cada peça já existia isolada, e nenhum trabalho encontrado juntava todas num formato de arquivo. E o ganho sem perdas de um modelo isolado tem teto, perto de 30%; o desperdício grande está em guardar o mesmo modelo várias vezes.

### 2. O formato

A ideia central virou "guardar só a surpresa": todo byte do arquivo é informação que o leitor não consegue prever com o que já tem. Três eixos de previsão: precisão (estratos), linhagem (pai e filho) e geração (blocos como receita). A regra da cadeia da entropia mostrou que as camadas saem de graça em teoria, e a analogia com vídeo (quadro-chave, quadro previsto, camadas escaláveis) organizou o desenho. Detalhes em [01-pesquisa.md](01-pesquisa.md).

### 3. Codec e primeiras medidas

O codec foi escrito em JavaScript puro, para rodar no navegador e no Node sem instalar nada ([`src/strata-core.js`](../src/strata-core.js)). Ele separa sinal, expoente e cada bit da mantissa em streams, comprime cada um, grava em ordem estrato-maior, lê qualquer prefixo e remonta o safetensors original byte a byte. Os testes de borda ([`tests/test-core.mjs`](../tests/test-core.mjs)) cobriram FP16, FP32, BF16, FP8, inteiros, zeros, infinitos e NaN.

As primeiras medidas confirmaram as previsões: arquivo com 70% do tamanho, volta idêntica, expoente com 2,7 bits de informação, mantissa quase aleatória. A genealogia simulada trouxe a descoberta que mudou o desenho: truncar o filho apaga o fine-tune, então o que se estratifica é a diferença, e não o peso.

### 4. Documento da pesquisa

A pesquisa foi escrita como documento do Claude, com o diagrama da estrutura e o gráfico de taxa × distorção feitos a partir dos números medidos. [01-pesquisa.md](01-pesquisa.md) é a transcrição dele, com os números atualizados.

### 5. Briefing do protótipo

Antes de desenhar a página, sete perguntas fecharam o escopo:

| Pergunta | Resposta |
| --- | --- |
| Quem vai ver? | Plateia leiga ou investidores |
| O que o protótipo precisa provar primeiro? | Que o arquivo volta bit a bit |
| Usar um modelo de verdade? | Não, um modelo sintético basta |
| Com o que a pessoa deve sair? | "Funciona de verdade" |
| Onde ela vai ver? | No celular, abrindo um link recebido |
| O que pareceria errado? | Promessa sem prova |
| Qual direção visual? | Painel de azulejos |

O contexto de produto ficou em [`PRODUCT.md`](../PRODUCT.md), e o contrato de direção em [`.impeccable/surfaces/strata-lab-html.md`](../.impeccable/surfaces/strata-lab-html.md).

### 6. Direção visual

O plugin sorteou um conjunto de mundos visuais (semente `904d49d5`, registrada em [`.impeccable/questions/roll.json`](../.impeccable/questions/roll.json)). Entre as propostas apresentadas (chapas de separação de gráfica, um cartório, o padrão da categoria e um painel de azulejos de estação), a escolhida foi o painel de azulejos.

A metáfora fechou com o formato: cada bloco de pesos é um azulejo, cada estrato é uma demão de cobalto, e a prova é o painel que volta idêntico. Ela recusa a demonstração de IA padrão, de fundo escuro, gradiente e cartões de indicadores.

### 7. Protótipo, versão 1

Página única com o pipeline num Web Worker, o modelo gerado no aparelho, a conversão de ida e volta, os 64 caracteres do SHA-256 conferidos azulejo a azulejo até o selo IDÊNTICO, o controle de demãos, a família e as paredes de 1 GB por azulejo. O detector de design do plugin apontou contraste baixo em textos e nos azulejos do SHA-256; os tons foram divididos em faixas claras e escuras até o detector não apontar mais nada.

### 8. Revisão de design, rodada 1

Uma revisão independente comparou a página com o contrato de direção e pediu oito correções materiais:

1. Azulejar a página inteira e dar ao computador uma composição própria, não uma coluna de celular no meio do cinza.
2. Trocar o fundo branco frio pelo branco-leite vidrado do mundo escolhido.
3. Pintar frisos, molduras e cartelas com traço de pincel de verdade, não com linhas vetoriais e sombras de CSS.
4. Pintar a albarrada demão a demão durante a conversão, em vez de mostrar uma lista de etapas.
5. Fazer o controle de demãos repintar o friso também, tornar visível a diferença entre 2 e 8 demãos e desenhar o trilho como 8 azulejos.
6. Mostrar a linha de ≈13 GB como estimativa e tirar o link para a pesquisa privada.
7. Não inverter as cores no tema escuro: o mesmo vidrado, com a luz baixa.
8. Trocar o triângulo padrão do `<summary>` e as etiquetas em pílula por formas do próprio mundo.

E pediu para manter três coisas: a conferência do SHA-256 azulejo a azulejo, a albarrada pintada com os pesos do próprio modelo e as paredes de 1 GB por azulejo.

### 9. Versão 2: o pincel

A resposta à correção 3 virou um módulo novo, [`src/strata-pincel.js`](../src/strata-pincel.js): ele pinta na abertura da página, no tamanho real do azulejo e nas cores do tema, os azulejos do fundo, o friso (pintado e só no risco), as cantoneiras, o florão, as volutas, as molduras e cartelas (em 9 fatias, via `border-image`), as placas dos botões e o seletor. O traço engorda e afina como pincel, a aguada tem manchas e acúmulo de tinta na borda.

As outras sete correções entraram junto:

- a página inteira azulejada, com 14 colunas a partir de 960 px;
- o vidrado branco-leite (`#EEE7D7`);
- a albarrada pintada durante a conversão, com cada demão decodificada de um prefixo do arquivo recém-escrito;
- o friso de 8 azulejos como controle, com uma lupa 3× sobre o vaso;
- a estimativa hachurada;
- o tema "meia-luz";
- o chevron desenhado e as fichas quadradas.

Também foram corrigidos título quebrando mal, botão quebrando em duas linhas, letreiro sobreposto, cantoneiras colidindo, cartelas estreitas no computador, a geometria dos arcos das cartelas e o encaixe da primeira tela em tablets.

### 10. Revisão de design, rodada 2

Sete das oito correções foram dadas como resolvidas; a composição do controle de demãos no computador ficou parcial. A revisão também achou duas regressões criadas pela própria correção:

- **O rejunte riscava o texto.** A camada de rejunte passava por cima das letras, e frases inteiras pareciam tachadas, inclusive a linha "Voltou idêntico." no tema escuro.
- **O rejunte cortava os azulejos de dados.** Ele não coincidia com as paredes de 1 GB, e um azulejo de dado podia aparecer partido em dois.

As correções: o rejunte foi para uma camada abaixo dos textos e dos azulejos de dados, que ganharam um halo do próprio vidrado; as paredes de dados foram encaixadas na grade do rejunte; e o painel de demãos no computador foi recomposto em duas colunas (a obra com o trilho, e ao lado a lupa e o texto).

### 11. Sistema de design documentado

Um documentador independente registrou o sistema a partir da página entregue, e não das intenções: [`DESIGN.md`](../DESIGN.md) e [`.impeccable/design.json`](../.impeccable/design.json), com paleta, tipos, grade de azulejos, camadas, movimento, componentes e regras nomeadas. Ele apontou setas em forma de caractere nos rótulos do controle de demãos, proibidas pelo padrão do plugin; elas foram retiradas. O detector apontou um texto pequeno demais e tamanhos fora da escala tipográfica; foram ajustados até o detector não apontar mais nada.

### 12. Números exatos

A primeira versão media o erro numa amostra de cerca de 390 mil pesos: E8M1 com 10,8% contra 14,3% do INT4. Como o modelo de teste tem só 1 milhão de pesos, a medida passou a usar todos eles: 10,95% contra 14,48%. A pesquisa, o gráfico, o `PRODUCT.md` e a página foram atualizados com os valores exatos, e o rótulo do INT8 foi corrigido para 8,5 bits por peso.

### 13. Publicação

A página foi publicada como página do Claude, privada até ser compartilhada, com o nome "Painel STRATA". Antes disso foi conferida no Chromium:

- celular claro e escuro;
- larguras de 320 a 1920 px, sem rolagem lateral;
- toques e teclado no controle de demãos;
- o caminho sem Web Worker e o caminho com o Worker bloqueado por política de segurança.

### 14. Este repositório

Tudo foi reunido aqui: o código-fonte separado da página montada, um script de montagem, o teste do pipeline completo (`tests/test-lab.mjs`), a especificação do formato, as fórmulas, os resultados, as capturas e este histórico. O arquivo em `dist/` é idêntico, byte a byte, à página publicada.

### 15. Segurança do leitor

Pergunta do Cantyll: o `.strata` mantém a segurança do `.safetensors`? Um fuzzer de mutação mostrou que o leitor original aceitava 18,9% dos arquivos adulterados devolvendo pesos errados sem erro, estourava erros de JavaScript em 22,9% e alocava gigabytes em 6,7% (bombas de rANS e deflate). Foram testados dois caminhos: endurecer o leitor JS e escrever um leitor de referência novo, em Rust, a partir da especificação. Os dois aplicam as mesmas regras (A.12) e os mesmos hashes (A.13), e o fuzzing diferencial deu zero divergência em 90 mil casos. Resultado na [seção 18 dos resultados](05-resultados.md#18-segurança-do-leitor).

Com o Rust escolhido, veio o leitor em Python (`leitor-py/`, binding com PyO3), com a API do safetensors e tensores do numpy e do torch sem cópia. Para ele o leitor passou a decodificar em várias threads, dividindo também o tensor de embedding, e a dizer pelo cabeçalho quantos bytes baixar para cada profundidade. A medição deixou claro onde o formato ganha: só quando a rede é o gargalo (abaixo de cerca de 190 Mbit/s numa máquina de 4 núcleos); de um disco local, o safetensors carrega na hora e o `.strata` precisa decodificar. Para empurrar esse limite, o leitor passou a decodificar enquanto baixa: cada stream é conferido e descomprimido assim que chega, e cada tensor é montado quando chega o último plano dele. Fundir cada plano no peso assim que ele chegasse foi tentado antes e descartado: zerava a espera no fim, mas passava 9 vezes pela saída e custava 2,5 vezes mais CPU. Com rodadas alternadas, o empate com o safetensors ficou entre 320 e 500 Mbit/s, e a 100 Mbit/s o modelo fica pronto 67 s antes.

## Decisões e por quê

| Decisão | Por quê |
| --- | --- |
| Atacar variantes e linhagem, não só compressão | Sem perdas, um modelo BF16 isolado não cai muito abaixo de 67%; o desperdício grande está nas cópias quase iguais |
| Implementar primeiro o modo FP-progressivo (bits da mantissa), e não o residual | É exato por construção e funciona com qualquer tipo de ponto flutuante; o residual ficou como extensão (B.1 da especificação) |
| Ordem estrato-maior no arquivo | Qualquer prefixo vira um modelo inteiro: download parcial por HTTP Range e modelo que melhora enquanto baixa |
| Reconstruir no ponto médio | Erra metade do que truncar erraria, e zero continua zero |
| Deflate do navegador como codificador | Roda em qualquer navegador sem biblioteca; custa 0,47 bit por peso, e o rANS ficou proposto |
| Cabeçalho original guardado dentro do `.strata` | Garante a volta bit a bit, inclusive espaços de alinhamento e metadados |
| Estratificar a diferença do fine-tune, e não o peso | Cosseno de 0,27 a 0,51 truncando o filho, contra 0,71 a 0,93 com um Δ de 1 bit |
| Modelo sintético, gerado no aparelho | Resposta do briefing, e tudo precisava rodar no navegador, sem instalar nada |
| Erro por tensor, com média ponderada | Evita que um tensor pequeno de valores grandes (a normalização) domine a conta |
| Medir com todos os pesos, sem amostra | No modelo de teste cabe, e a amostra deslocava os números visíveis na página |
| Painel de azulejos | Escolha no briefing; a metáfora de demãos é o próprio formato |
| Estimativas sempre marcadas | "Promessa sem prova" foi a resposta para o que pareceria errado |
| Leitor que recusa em vez de adivinhar | Um campo fora da regra recusa o arquivo inteiro; aceitar "o que dá para ler" foi o que deixava 18,9% dos arquivos adulterados passarem |
| Inflate próprio no JS, não o `DecompressionStream` | Para no tamanho declarado, recusa bytes sobrando (o do Node aceita, o dos navegadores não) e foi mais rápido no Node |
| Hash por stream e por tensor, não árvore binária | Com uma folha por stream o cabeçalho cresce 0,0064%; a árvore só compensa com blocos de 4.096 pesos (B.2) |
| Rust como leitor de referência, JS só na página | Segurança empatou; o Rust é 4 a 5 vezes mais rápido e é a base para Python e carregadores |
| Leitor em Python com a API do safetensors, sem cópia | Quem carrega modelos já chama `load_file` e `safe_open`; trocar a importação é o menor passo para testar o formato |
| Decodificar enquanto baixa, com as mesmas recusas | Só ganha onde a rede é o gargalo, que é exatamente onde o formato quer servir; o limite de saída é aplicado sobre os bytes já recebidos |
| Montar cada tensor quando chega o último plano dele, não fundir plano a plano | As duas zeram a espera depois do último byte; fundir plano a plano passa 9 vezes pela saída e custa 2,5 vezes mais CPU |
| Kernel de inferência em C/C++/CUDA, não em Rust | O kernel mora dentro do llama.cpp e do ggml; o Rust fica antes dele: valida, decodifica, converte e alimenta os carregadores |

## O que ficou em aberto

- **Medir num modelo real**, com perplexidade, contra Q4_K_M e Q8_0 (fase 2 do [roteiro de validação](01-pesquisa.md#roteiro-de-validação)).
- **A Parte B da especificação**: modo residual, codec por bloco, genealogia gravada no arquivo, árvore Merkle por bloco e codificador rANS ([02-especificacao-formato.md](02-especificacao-formato.md#parte-b--extensões-propostas-ainda-sem-código)).
- **Nome comercial** e um **link público da pesquisa**: o documento original continua privado.
