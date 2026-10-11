# STRATA

**Um formato de arquivo para pesos de IA que guarda só o que não dá para prever.** Cada estrato refina o anterior, cada modelo-filho guarda só a diferença para o pai, e o arquivo volta bit a bit para o `.safetensors` original.

![Cortar o .strata em qualquer estrato devolve um modelo válido](docs/img/diagrama-estrutura.png)

## A ideia em três linhas

- **Um arquivo, todas as precisões.** Os bits de cada peso viram camadas em ordem de importância. Ler só o começo do arquivo dá um modelo de 4 a 7 bits; ler tudo dá o original exato. Hoje, cada precisão é um arquivo separado.
- **Uma família, não cópias.** Um fine-tune aponta o modelo-base e guarda só a diferença.
- **Sem perdas por padrão.** A conversão a partir do `.safetensors` é direta e volta byte a byte, conferida pelo SHA-256.

O ganho grande não vem de espremer um modelo (sem perdas, o teto fica perto de 30%), e sim de parar de guardar o mesmo modelo várias vezes: um único repositório do Hugging Face guarda 29 versões do mesmo modelo de 9 bilhões de pesos, somando 191 GB.

## O que o protótipo provou

Medido num modelo sintético BF16 de 1.000.192 pesos (semente 7), com o código deste repositório:

| | Resultado |
| --- | --- |
| Volta para `.safetensors` | SHA-256 idêntico, byte a byte |
| Tamanho do `.strata` | 70,0% do original (o teto teórico fica perto de 67%) |
| Lendo só 46,5% do arquivo (E8M1) | erro de 10,95%, contra 14,48% de um INT4 por bloco |
| Fine-tune guardado como diferença do pai | 24,7% do tamanho do filho guardado sozinho |
| Informação no expoente | 2,70 bits dos 8 que ele ocupa |

Os pesos são sintéticos (caudas pesadas e canais atípicos, como num LLM), então os números servem para comparar opções, não para prometer ganhos. Tabelas completas, outras sementes e um modelo 4× maior em [docs/05-resultados.md](docs/05-resultados.md).

## Experimente

**A página.** [`dist/strata-lab.html`](dist/strata-lab.html) é um arquivo só, sem servidor. Baixe e abra no navegador: ela gera um modelo de teste no aparelho, converte para `.strata`, volta para `.safetensors` e confere o SHA-256, pintando o resultado num painel de azulejos. Nada sai do aparelho.

<p>
  <img src="docs/img/prototipo-celular.png" alt="Primeira tela no celular: O arquivo volta idêntico" width="260">
  <img src="docs/img/prototipo-conversao.png" alt="Conversão em andamento, pintada demão a demão" width="260">
</p>

**Os testes** (Node 18 ou mais novo):

```
node tests/test-lab.mjs     # o pipeline da página: tabela por estrato e seis conferências
node tests/test-core.mjs    # o codec: modelos de 1 M e 4 M pesos, prefixos e bordas de tipos
python3 build.py            # monta dist/strata-lab.html a partir de src/
```

Dá para rodar tudo pelo navegador, sem instalar nada no computador: abra o repositório num Codespace do GitHub e use o terminal dele.

## Documentação

| Documento | Para quem quer |
| --- | --- |
| [01 · Pesquisa](docs/01-pesquisa.md) | entender o problema, o estado da arte e a proposta |
| [02 · Especificação do formato](docs/02-especificacao-formato.md) | ler ou escrever um `.strata`: layout binário, cabeçalho, prefixos, exemplo byte a byte e extensões propostas |
| [03 · Fórmulas](docs/03-formulas.md) | conferir a matemática: regra da cadeia, custo por estrato, ponto médio, métricas, genealogia |
| [04 · Protótipo](docs/04-prototipo.md) | saber como a página funciona e como foi desenhada |
| [05 · Resultados](docs/05-resultados.md) | ver todos os números medidos e o que eles não provam |
| [06 · Histórico](docs/06-historico.md) | acompanhar como tudo foi feito, as revisões e as decisões |
| [DESIGN.md](DESIGN.md) | o sistema visual do painel de azulejos |
| [PRODUCT.md](PRODUCT.md) | o contexto de produto do protótipo |

## O que tem aqui

```
docs/                     pesquisa, especificação, fórmulas, protótipo, resultados, histórico
docs/img/                 diagrama, gráfico e capturas do protótipo
src/strata-core.js        codec: separa os campos, monta e lê o .strata, mede o erro
src/strata-lab.js         pipeline da demonstração (roda num Web Worker)
src/strata-pincel.js      pinta na hora os azulejos, frisos, molduras e botões da página
src/strata-lab.src.html   a página, antes da montagem
dist/strata-lab.html      a página montada, pronta para abrir (≈127 KB)
build.py                  injeta os scripts na página e gera dist/
tests/                    testes do codec e do pipeline completo
DESIGN.md, .impeccable/   sistema de design e contrato de direção
PRODUCT.md                contexto de produto
```

## Estado

- **Implementado (`strata/0.1`):** estratos de sinal, expoente e cada bit da mantissa em ordem estrato-maior; leitura por prefixo; reconstrução no ponto médio; volta bit a bit para `.safetensors`; BF16, FP16, FP32, FP8 e inteiros.
- **Medido, mas ainda não gravado no arquivo:** a genealogia (o filho como diferença do pai).
- **Proposto, sem código:** modo residual sobre um quantizador de ponta, codec por bloco, árvore Merkle, codificador rANS para GPU e um adaptador que apresenta o `.strata` como `.safetensors`.
- **Próximo passo:** medir num modelo real (Qwen3-0.6B ou Llama-3.2-1B), com perplexidade, contra Q4_K_M e Q8_0. O roteiro completo está na [pesquisa](docs/01-pesquisa.md#roteiro-de-validação).

## Limites

- Os pesos medidos são sintéticos, e o erro é medido nos pesos, não na qualidade das respostas do modelo.
- O codificador é o deflate do navegador, que gasta cerca de 0,5 bit por peso a mais que o limite de entropia.
- Perto de 8 bits, o INT8 por bloco erra menos que o corte equivalente do STRATA; o modo residual foi proposto para fechar essa diferença.
- A linha "≈13 GB num `.strata`" da página é uma estimativa e aparece marcada como tal.

---

Pesquisa, formato e protótipo criados por Cantyll com o Claude, em outubro de 2026. Fontes da pesquisa em [docs/01-pesquisa.md](docs/01-pesquisa.md#fontes).
