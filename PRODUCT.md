# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

HTML estático em um único arquivo, sem build (pedido do usuário: "protótipo em html"). Tudo roda no navegador: o usuário não pode instalar nada localmente no computador de trabalho.

## Users

Plateia leiga ou investidores (confirmado). Assistem ou manipulam uma demonstração do formato STRATA sem conhecer quantização, safetensors ou teoria da informação.

## Product Purpose

STRATA é um conceito de formato de armazenamento para pesos de modelos de IA grandes: estratos de bits em ordem de importância (um arquivo, todas as precisões), genealogia entre modelos (fine-tune guardado como delta do pai) e conversão direta e reversível a partir de safetensors/GGUF. O protótipo existe para provar, primeiro, a volta bit-exata: safetensors → .strata → safetensors com SHA-256 idêntico (confirmado como prioridade).

## Positioning

Formatos atuais guardam números numa precisão fixa, um arquivo por variante e cada modelo isolado. STRATA guarda só a informação que ainda não dá para prever: precisão vira escolha de leitura, fine-tunes viram deltas, e o original volta byte a byte.

## Operating Context

Demonstração aberta num navegador (celular ou computador). O usuário trabalha só pelo navegador; nada local.

## Capabilities and Constraints

- O codec do protótipo é real: separa sinal, expoente e mantissa; comprime com o deflate do navegador (CompressionStream); reconstrói e confere SHA-256.
- Fonte de dados: modelo sintético gerado na página (confirmado: sintético basta). Upload de .safetensors é opcional, não exigido.
- Números medidos (modelo sintético BF16 de 1 M pesos, semente 7): .strata = 70,0% do safetensors; expoente com 2,70 bits de entropia; E8M1 com 10,9% de erro relativo (RMS, média por tensor) contra 14,5% do INT4 por bloco de 32; filho de fine-tune como delta = 24,7% do filho isolado.
- Na página, o modelo leva junto o desenho de demonstração (43.200 pesos), que comprime melhor: lá o .strata fica em 68,8% e o delta do filho em 24,1%. O erro mostrado exclui o desenho.
- Indefinido: nome comercial, marca, preço, modelo de negócio, parceiros.

## Brand Commitments

Nome de trabalho: STRATA (extensão .strata). Nenhum ativo de marca existe.

## Evidence on Hand

- Pesquisa no Claude Docs: "STRATA — formato estratificado e genealógico para pesos de IA".
- Núcleo do codec testado: `strata-core.js` (round-trip bit-exato, propriedade de prefixo, bordas FP16/FP32/FP8/I64).
- Não existem: clientes, depoimentos, benchmarks em modelos reais, parcerias. Não inventar.

## Product Principles

1. Provar antes de afirmar: todo número na tela sai de uma medição feita ali mesmo.
2. Exato é inegociável: a volta bit a bit é a promessa central.
3. Sem jargão no primeiro contato; detalhe técnico só para quem pede.
4. Honestidade sobre limites: sem perdas, o teto num modelo isolado é ~30%.
