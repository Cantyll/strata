# Resultados

Tudo o que o protótipo mediu, com os números exatos, de onde eles vêm e como repetir a conta. Medido em 10 de outubro de 2026.

Há duas configurações, e os documentos citam as duas:

- **Modelo de teste:** o modelo sintético BF16 de 1.000.192 pesos, semente 7. É o que `tests/test-lab.mjs` roda e o que a pesquisa e as fórmulas citam.
- **Página:** o mesmo modelo mais o desenho de demonstração (43.200 pesos), que é o que [`dist/strata-lab.html`](../dist/strata-lab.html) converte. O desenho comprime melhor que os pesos, então os tamanhos da página saem um pouco menores. O erro mostrado na página exclui o desenho.

Para repetir: `node tests/test-lab.mjs` (Node 18 ou mais novo). Leva cerca de 1 segundo.

> **Formato 0.2 (15 de outubro de 2026).** As seções 1 a 9 foram medidas com o formato 0.1, que usava deflate em todos os streams. Desde a 0.2 o expoente usa rANS e o modelo de teste cai de 70,0% para **67,1%** do original (E0 = 3,71 bits/peso, igual à coluna "Limite de entropia" da seção 3). O efeito num modelo real está na [seção 12](#12-rans-no-expoente-num-modelo-real-llama-32-1b).

## 1. Resumo

| Medida | Modelo de teste | Página |
| --- | --- | --- |
| Pesos | 1.000.192 | 1.043.392 |
| `.safetensors` original | 2.001.080 bytes | 2.087.576 bytes (2,09 MB) |
| `.strata` | 1.401.011 bytes | 1.436.827 bytes (1,44 MB) |
| Tamanho do `.strata` | **70,0%** do original | **68,8%** (31,2% menor) |
| Volta para `.safetensors` | SHA-256 idêntico | SHA-256 idêntico, 16.700.608 bits conferidos um a um |
| Erro lendo 2 de 8 estratos (E8M1) | **10,95%**, contra 14,48% do INT4 por bloco | o mesmo (o erro exclui o desenho) |
| Filho de fine-tune guardado como diferença | **24,7%** do filho guardado sozinho | 24,1% |
| Informação no expoente | 2,70 bits dos 8 que ele ocupa | 2,70 bits |

SHA-256 do original e da volta:

- Modelo de teste: `8b2d1893e913dfe60455d9d6a63bbc5e9a9446d88d26eda52bc8d8c2b65caa90`
- Página: `2bda844f6755e8d741956c5c49bc307ccb2883b951543b156c735c1ced939a02`

O gerador do modelo é determinístico (semente fixa), então esses resumos se repetem no Node e no Chromium (motor V8). Outro motor de JavaScript pode calcular `Math.log` e `Math.cos` com diferenças mínimas e gerar um modelo com outro resumo; a volta continua idêntica.

## 2. O modelo de teste

Cinco matrizes no formato de uma camada de LLM, mais uma normalização. Cada linha tem escala própria, 0,4% dos pesos são caudas pesadas (×6) e duas colunas por matriz são canais atípicos (×8). A receita está em [03-formulas.md](03-formulas.md#12-o-modelo-sintético).

| Tensor | Forma | Pesos | Escala (σ) | No fine-tune simulado |
| --- | --- | --- | --- | --- |
| `model.embed_tokens.weight` | 1172 × 256 | 300.032 | 0,020 | congelado |
| `model.layers.0.self_attn.q_proj.weight` | 469 × 256 | 120.064 | 0,018 | LoRA de posto 8 |
| `model.layers.0.self_attn.v_proj.weight` | 469 × 256 | 120.064 | 0,012 | LoRA de posto 8 |
| `model.layers.0.mlp.up_proj.weight` | 898 × 256 | 229.888 | 0,016 | ajuste completo |
| `model.layers.0.mlp.down_proj.weight` | 898 × 256 | 229.888 | 0,014 | congelado |
| `model.layers.0.input_layernorm.weight` | 256 | 256 | 1 + 0,05·g | ajuste completo |
| **Total** | | **1.000.192** | | |

O cabeçalho do `.strata` tem 5.172 bytes (0,37% do arquivo): a lista de tensores, a tabela de 54 streams e o cabeçalho original do safetensors guardado em base64.

## 3. Estrato por estrato

Cada linha é um corte do mesmo arquivo. "Bits por peso" é acumulado: quanto custa ler até aquele estrato.

| Estrato | Equivale a | Bytes gravados | Bits por peso (deflate) | Limite de entropia | Prefixo do `.strata` | Fração do original | Erro relativo | SNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | sinal + expoente | 522.353 | 4,18 | 3,70 | 527.537 B · 37,7% | 26,4% | 23,17% | 12,7 dB |
| E1 | E8M1 | 123.330 | 5,16 | 4,68 | 650.867 B · 46,5% | 32,5% | 10,95% | 19,2 dB |
| E2 | E8M2 | 125.024 | 6,16 | 5,68 | 775.891 B · 55,4% | 38,8% | 5,40% | 25,4 dB |
| E3 | E8M3 | 125.024 | 7,16 | 6,68 | 900.915 B · 64,3% | 45,0% | 2,67% | 31,5 dB |
| E4 | E8M4 | 125.024 | 8,16 | 7,68 | 1.025.939 B · 73,2% | 51,3% | 1,35% | 37,4 dB |
| E5 | E8M5 | 125.024 | 9,16 | 8,68 | 1.150.963 B · 82,2% | 57,5% | 0,71% | 43,0 dB |
| E6 | E8M6 | 125.024 | 10,16 | 9,68 | 1.275.987 B · 91,1% | 63,8% | 0,41% | 47,8 dB |
| E7 | BF16 exato | 125.024 | 11,16 | 10,68 | 1.401.011 B · 100% | 70,0% | zero | exato |

- **Prefixo** inclui os 12 bytes iniciais e o cabeçalho: é exatamente o que um leitor precisa baixar para aquela profundidade.
- **Limite de entropia** é o que um codificador de entropia ideal gastaria (seção 4).
- **Erro relativo** é o RMS do erro sobre o RMS dos pesos, por tensor, com média ponderada pelo tamanho do tensor ([03-formulas.md](03-formulas.md#8-métrica-de-erro-usada)). SNR = −20·log₁₀(erro).
- A cada estrato o erro cai para 0,47 a 0,57 do anterior, como a fórmula do ponto médio prevê ([03-formulas.md](03-formulas.md#7-cada-estrato-divide-o-erro-por-dois)).

## 4. O que cada parte custa

| Parte | Bytes crus | Bytes gravados | Bits por peso gravados | Entropia medida | Como foi gravada |
| --- | --- | --- | --- | --- | --- |
| Sinal | 125.024 | 124.997 | 1,00 | 1,00 | cru em 5 tensores; deflate só na normalização, de pesos todos positivos |
| Expoente | 1.000.192 | 397.356 | 3,18 | 2,70 | deflate nos 6 tensores |
| Mantissa, bit 1 | 125.024 | 123.330 | 0,99 | 0,98 | deflate em 5 tensores; cru na normalização |
| Mantissa, bit 2 | 125.024 | 125.024 | 1,00 | 0,99 | cru: o deflate aumentaria |
| Mantissa, bits 3 a 7 | 125.024 cada | 125.024 cada | 1,00 cada | 1,00 cada | cru |

O que a tabela mostra:

- **Só o expoente comprime de verdade.** Ele ocupa 8 bits por peso, mas o modelo inteiro usa só 28 dos 256 valores possíveis: 4 valores cobrem 80% dos pesos e 8 cobrem 97%. No tensor do vocabulário, os 4 bits de cima do expoente quase nunca mudam (entropia de 0,00 a 0,01 bit cada).
- **O deflate desperdiça 0,47 bit por peso**, todo no expoente (3,18 gravados contra 2,70 de entropia). É a diferença entre a curva cheia e a tracejada no gráfico da pesquisa, e o motivo de a especificação propor rANS ([02-especificacao-formato.md](02-especificacao-formato.md#b5-codificador)).
- **A mantissa é praticamente aleatória.** Do 2º ao 7º bit, o deflate não ganha nada e o codificador grava cru. É por isso que nenhum formato sem perdas tira muito mais de 30% de um modelo BF16.

Entropia aqui é a de ordem zero: histograma de bytes para o expoente e entropia binária da fração de uns para o sinal e para cada bit da mantissa, com média ponderada pelo tamanho do tensor. Um codificador que use contexto pode ficar um pouco abaixo dela.

## 5. Contra a quantização de hoje

Referências calculadas nos mesmos pesos: quantização simétrica por bloco de 32, com escala de 16 bits ([03-formulas.md](03-formulas.md#9-referências-de-quantização-por-bloco)).

| Opção | Bits por peso | Erro relativo | Onde está |
| --- | --- | --- | --- |
| INT4 por bloco de 32 | 4,5 | 14,48% | um arquivo só para ela |
| STRATA, 2 demãos (E8M1) | 5,16 (deflate) · 4,68 (limite) | 10,95% | os primeiros 46,5% do `.strata` |
| STRATA, 3 demãos (E8M2) | 6,16 · 5,68 | 5,40% | os primeiros 55,4% |
| STRATA, 4 demãos (E8M3) | 7,16 · 6,68 | 2,67% | os primeiros 64,3% |
| STRATA, 5 demãos (E8M4) | 8,16 · 7,68 | 1,35% | os primeiros 73,2% |
| INT8 por bloco de 32 | 8,5 | 0,83% | um arquivo só para ela |
| STRATA, 6 demãos (E8M5) | 9,16 · 8,68 | 0,71% | os primeiros 82,2% |

A leitura honesta:

- **Perto de 4 bits, o STRATA erra menos.** O corte E8M1 erra 10,95% contra 14,48% do INT4, pagando 0,2 bit por peso a mais no limite de entropia e 0,7 com o deflate.
- **Perto de 8 bits, o INT8 por bloco ganha.** Ele erra 0,83% com 8,5 bits; o STRATA erra 1,35% com 8,16 bits e só passa à frente (0,71%) com 9,16. Truncar a mantissa não é o quantizador mais eficiente em cada taxa; o modo residual da especificação ([B.1](02-especificacao-formato.md#b1-modo-residual)) existe para fechar essa diferença.
- **A vantagem não está em vencer cada quantizador na sua taxa.** Está em ter todas as taxas, mais o original exato, num arquivo só.

## 6. Outras sementes e um modelo 4× maior

Para conferir que a semente 7 não é um caso feliz, o mesmo pipeline rodou com outras sementes:

| Semente | `.strata` / original | Erro E8M1 | Erro INT4 | Filho como diferença | Entropia do expoente | Volta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 70,07% | 11,11% | 15,24% | 24,78% | 2,709 | idêntica |
| 2 | 70,06% | 10,95% | 14,98% | 24,75% | 2,711 | idêntica |
| 3 | 70,05% | 10,87% | 15,03% | 24,79% | 2,708 | idêntica |
| **7 (padrão)** | **70,01%** | **10,95%** | **14,48%** | **24,75%** | **2,705** | **idêntica** |
| 8 | 70,06% | 10,95% | 15,27% | 24,78% | 2,710 | idêntica |
| 42 | 70,05% | 11,05% | 15,24% | 24,65% | 2,710 | idêntica |

Em todas, E8M1 erra menos que o INT4. A semente 7 dá o INT4 mais favorável da lista, então a vantagem de E8M1 mostrada na página é a menor entre essas sementes.

**Modelo de 4.001.280 pesos** (semente 7, matrizes de 512 colunas, `tests/test-core.mjs`): 8.003.264 → 5.574.146 bytes (69,6%), volta com SHA-256 idêntico, E8M1 com 10,99% de erro contra 13,88% do INT4. Aqui as métricas usam uma amostra de cerca de 2 milhões de pesos, em segmentos contíguos de 4.096.

## 7. Genealogia: o filho guardado como diferença

O fine-tune é simulado (receita em [03-formulas.md](03-formulas.md#11-o-fine-tune-simulado)). Cada tensor do filho vira `HERDA` (idêntico ao do pai, 32 bytes de referência) ou `DELTA` (o XOR com o pai, estratificado e comprimido).

| Tensor | Ajuste | Modo | Pesos alterados | Filho sozinho | Como diferença | Proporção | Cosseno, E8M3 | Cosseno, Δ de 1 bit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vocabulário | congelado | `HERDA` | 0% | 418.594 B | 32 B | 0,01% | — | — |
| atenção q | LoRA r=8 | `DELTA` | 86,2% | 167.901 B | 103.782 B | 61,8% | 0,51 | 0,84 |
| atenção v | LoRA r=8 | `DELTA` | 85,6% | 167.855 B | 102.442 B | 61,0% | 0,50 | 0,84 |
| MLP up | completo | `DELTA` | 67,5% | 320.450 B | 138.900 B | 43,3% | 0,27 | 0,71 |
| MLP down | congelado | `HERDA` | 0% | 320.758 B | 32 B | 0,01% | — | — |
| normalização | completo | `DELTA` | 61,3% | 297 B | 234 B | 78,8% | 0,35 | 0,93 |
| **Total** | | | | **1.395.855 B** | **345.422 B** | **24,7%** | | |

- **"Filho sozinho"** é o custo do filho comprimido sem pai, com o mesmo codificador; **"pesos alterados"** é a fração de palavras BF16 diferentes das do pai.
- **O LoRA mexe em 86% dos pesos.** O tamanho do ajuste dele não depende do peso (o RMS é 1% do desvio do tensor), então nos pesos pequenos ele alcança os bits altos da mantissa. O ajuste completo simulado é proporcional ao peso (cerca de 0,4%), menor que o passo do BF16 em boa parte dos pesos: um terço deles nem muda.
- **Cossenos:** comparam a diferença real entre filho e pai com o que sobra dela em duas aproximações. Truncar os dois modelos em E8M3 preserva pouco da direção do ajuste (0,27 a 0,51); um Δ de 1 bit por peso, com escala por bloco de 64, preserva bem mais (0,71 a 0,93). É a evidência para estratificar o Δ, e não o peso. No tensor MLP up, maior que o orçamento, o cosseno usa uma amostra de 131.072 pesos.
- **Na página**, o desenho entra como mais um tensor congelado (`HERDA`): 345.454 de 1.430.869 bytes, **24,1%**.

## 8. O que a página mostra

A página converte o modelo com o desenho e mostra, em cada seção, números medidos naquele momento.

**A prova:** original 2,09 MB, `.strata` 1,44 MB, "31,2% menor, sem perder um bit"; os 64 caracteres do SHA-256 conferidos par a par até "64 de 64 azulejos batem"; "16.700.608 bits conferidos, um por um".

**Quanto ler é escolha sua** (celular de 390 px de largura):

| Demãos lidas | Estrato | Lido do `.strata` | Erro | Tons na lupa | Tons no desenho inteiro | Frase mostrada |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | E0 | 36,9% | 23,17% | 3 | 3 | "Só o esboço: sinal e escala de cada peso. A forma já está lá, o detalhe ainda não." |
| 2 | E1 | 45,6% | 10,95% | 5 | 5 | "Erro na faixa de um modelo compactado em 4 bits, que neste mesmo teste erra 14,5%." |
| 3 | E2 | 54,6% | 5,40% | 10 | 10 | "Detalhe médio, com pouco mais da metade do arquivo." |
| 4 | E3 | 63,7% | 2,67% | 17 | 18 | "Erro abaixo de 3,0%: no desenho, a diferença some a olho nu." |
| 5 | E4 | 72,8% | 1,35% | 31 | 34 | "Erro abaixo de 2,0%: …" |
| 6 | E5 | 81,8% | 0,71% | 55 | 66 | "Erro abaixo de 1,0%: …" |
| 7 | E6 | 90,9% | 0,41% | 101 | 130 | "Erro abaixo de 1,0%: …" |
| 8 | E7 | 100% | zero | 188 | 258 | "Original exato, bit a bit. É o arquivo inteiro." |

- O desenho ocupa três oitavas de expoente, então só o estrato 0 já tem 3 tons, e cada bit de mantissa quase dobra a contagem.
- A lupa conta os tons num recorte de 30 linhas do vaso; a largura do recorte segue a proporção da tela, então a contagem muda um pouco em outros aparelhos. A contagem do desenho inteiro não muda.

**Um ajuste guarda só o que mudou:** filho sozinho 1,43 MB, como diferença do pai 0,35 MB (24,1%); por parte: vocabulário 0 KB, atenção 104 KB e 102 KB, memória (MLP) 139 KB e 0 KB, ajuste de escala menos de 1 KB, desenho 0 KB.

**Hoje, um modelo vira 29 arquivos:** 191 GB nos 29 arquivos do repositório gemma-2-9b-it-GGUF; 97 GB com a deduplicação do Hugging Face; ≈13 GB num `.strata` com todas as precisões e o original. A última linha é estimativa (9,24 bilhões de pesos × 11,0 bits por peso, a média medida na página) e aparece hachurada.

## 9. Tempos

Tempos de uma máquina só, um contêiner na nuvem; não foram medidos num celular de verdade.

| Etapa | O que faz | Node 22 | Chromium, na página |
| --- | --- | --- | --- |
| ler | lê o safetensors e separa os tensores | 0–1 ms | 3 ms |
| separar | desmonta cada número em sinal, expoente e bits de mantissa | 47–58 ms | 70 ms |
| comprimir | comprime cada stream e monta o `.strata` | 120–177 ms | 111 ms |
| camadas | decodifica o desenho de 8 prefixos (só na página) | — | 29 ms |
| remontar | decodifica tudo e remonta o safetensors | 76–104 ms | 89 ms |
| conferir | SHA-256 e comparação byte a byte | 5–15 ms | 17 ms |
| medir | erro por estrato, INT4, INT8 e entropia | 92–173 ms | 240 ms |
| família | simula o fine-tune e mede o filho como diferença | 249–396 ms | 390 ms |

- **Node 22:** gerar o modelo leva 60 a 76 ms, e a conversão completa (ida, volta, conferência e medidas) leva de 0,6 a 0,7 s depois do primeiro aquecimento.
- **Chromium:** do toque no botão ao resultado, 1,1 s, com o pipeline num Web Worker. Na tela, a conversão leva alguns segundos a mais por causa das animações das demãos e da prova.
- **Modelo de 4 M pesos:** ida em 0,76 s e volta em 0,31 a 0,33 s (Node 22).

## 10. O que estes números não provam

- **O modelo é sintético.** Caudas pesadas e canais atípicos imitam um LLM, mas não o substituem. Os trabalhos publicados com modelos BF16 reais ficam na mesma faixa (ZipNN: −33%; DFloat11: cerca de 70% do tamanho), mas um modelo real pode comprimir um pouco diferente.
- **Erro nos pesos não é qualidade do modelo.** Nenhuma perplexidade foi medida. O próximo passo do [roteiro de validação](01-pesquisa.md#roteiro-de-validação) é medir E8M1 a E8M7 num modelo real (Qwen3-0.6B ou Llama-3.2-1B) contra Q4_K_M e Q8_0.
- **Só BF16 passou de ponta a ponta na página.** FP16, FP32, FP8 (E4M3 e E5M2), inteiros, zeros, infinitos e NaN passam nos testes de borda do codec (`tests/test-core.mjs`).
- **A genealogia é simulada** e o delta é medido, não gravado num `.strata` (limite da v0.2, ver [A.10](02-especificacao-formato.md#a10-limites-conhecidos-da-v02)).
- **As seções 1 a 9 usam o codificador da 0.1** (deflate em tudo). Com o rANS da 0.2 os tamanhos ficam no limite de entropia da seção 3; ver a seção 12.
- **A linha de ≈13 GB é estimativa**, e os tempos são de uma máquina só.

## 11. Como repetir

```
node tests/test-lab.mjs     # pipeline da página, sem o desenho: tabela por estrato e seis conferências
node tests/test-core.mjs    # codec: modelos de 1 M e 4 M, prefixos, genealogia e bordas de tipos
python3 build.py            # gera dist/strata-lab.html a partir de src/
```

Os números da página (seção 8) foram lidos abrindo `dist/strata-lab.html` no Chromium, com tela de 390 × 844, tocando em "Converter e conferir" e passando pelas 8 posições do friso de demãos.

## 12. rANS no expoente num modelo real (Llama-3.2-1B)

Medido em 15 de outubro de 2026 em `unsloth/Llama-3.2-1B` (cópia aberta do `meta-llama/Llama-3.2-1B`, 1.235.814.400 pesos BF16, `model.safetensors` com SHA-256 `68a2e4be76fa7094…`). Para repetir:

```
node tests/test-rans.mjs model.safetensors --json saida.json   # ~9 min em 4 núcleos, um tensor por vez na memória
```

Os números por tensor estão em [`tests/resultados/rans__unsloth__Llama-3.2-1B.json`](../tests/resultados/rans__unsloth__Llama-3.2-1B.json).

**Volta bit a bit.** Os 146 tensores foram codificados com rANS no expoente, decodificados na profundidade total e remontados num `.safetensors`: SHA-256 idêntico ao original.

**O expoente**, nas 112 matrizes de `model.layers` (973.078.528 pesos, as mesmas que o ΔPPL corta):

| Codificador | Bits/peso |
| --- | ---: |
| Entropia de ordem 0, por tensor (o limite deste rANS) | 2,6071 |
| rANS da 0.2 | **2,6075** |
| Deflate do core da 0.1 (`CompressionStream`, Node) | 3,0616 |
| Deflate do `strata_ppl.py` (zlib do Python, a tabela do Colab) | ≈3,13 |

O rANS fica a 0,0004 bit/peso da entropia: a tabela de frequências e o estado final custam cerca de 100 bytes por tensor. A pior distância num tensor (0,14 bit/peso) é numa normalização de 2.048 pesos, onde a tabela pesa; nas matrizes grandes a distância some.

**Cada estrato fica 0,454 bit/peso menor** que no core da 0.1 (0,52 menor que na tabela do Colab, que usou o zlib do Python):

| Estrato | 0.1, deflate (b/peso) | 0.2, rANS (b/peso) | Fração do BF16 |
| --- | ---: | ---: | ---: |
| E0 | 4,062 | 3,608 | 22,5% |
| E1 | 5,048 | 4,594 | 28,7% |
| E2 | 6,048 | 5,594 | 35,0% |
| E3 | 7,048 | 6,594 | 41,2% |
| E4 | 8,048 | 7,594 | 47,5% |
| E7 (sem perdas) | 11,048 | **10,594** | **66,2%** |

No modelo inteiro (com embedding e normas) o estrato 7 dá 10,584 bits/peso, cerca de 1,64 GB contra 2,47 GB do safetensors. Isto põe o STRATA sem perdas na faixa que a literatura atribui ao DFloat11 e ao ZipNN (cerca de 11 bits, 70%). Não rodei nenhum dos dois neste modelo, então a comparação direta continua em aberto.

**Contra o INT-k, com o mesmo codificador dos dois lados.** O ΔPPL é o da tabela do Colab (`tests/ppl/resultados/`, wikitext-2 inteiro); os bits do STRATA são os da 0.2. A coluna "INT-k com entropia" mede a entropia de ordem 0 dos códigos INT-k por tensor mais 0,5 bit da escala: é o que o INT-k ocuparia se ganhasse o mesmo tipo de codificador que o expoente ganhou agora.

| Corte | STRATA 0.2 | ΔPPL | INT-k | nominal | com entropia | ΔPPL |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| E1 | 4,594 | +13,08% | INT4 | 4,5 | 4,100 | +16,84% |
| E2 | 5,594 | +2,74% | INT5 | 5,5 | 5,169 | +3,63% |
| E3 | 6,594 | +0,61% | INT6 | 6,5 | 6,189 | +0,68% |
| E4 | 7,594 | +0,15% | INT8 | 8,5 | 8,167 | +0,04% |

- Contra o INT-k **nominal**, o STRATA agora gasta 0,09 bit a mais e erra menos em E1, E2 e E3.
- Contra o INT-k **com entropia**, o STRATA ainda gasta 0,4 a 0,5 bit a mais pelo mesmo ΔPPL. A causa continua sendo a mesma: o expoente é pago em cada peso, e a escala do INT-k é paga uma vez a cada 32 pesos.
- Nenhum dos dois é o adversário de produção. Os K-quants do llama.cpp (Q4_K_M, Q5_K_M) não usam codificador de entropia, mas escolhem a escala minimizando o erro. Essa é a próxima comparação.

**Quanto ainda há para ganhar sem perdas.** Entropia condicional empírica nas mesmas 112 matrizes, já descontando o custo das tabelas e da informação lateral:

| Contexto | Antes (b/peso) | Depois (b/peso) | Ganho |
| --- | ---: | ---: | ---: |
| Expoente dado o expoente máximo da linha (8 bits por linha) | 2,607 | 2,592 | 0,015 |
| Expoente dado os máximos da linha e da coluna | 2,607 | 2,590 | 0,017 |
| Mantissa bit 1 dado o expoente | 0,986 (deflate) | 0,948 | 0,038 |
| Mantissa bit 2 dado o expoente | 1,000 | 0,986 | 0,014 |
| Mantissa bits 3 a 7 dado o expoente | 5,000 | 4,996 | 0,004 |

O contexto renderia, somado, cerca de 0,07 bit/peso: o estrato 7 iria de 10,59 para perto de 10,52. Condicionar a mantissa no expoente é algo que a ordem estrato-maior permite de graça, porque o expoente sempre chega antes. Mesmo assim, o ganho é pequeno. Superar o DFloat11 em tamanho sem perdas vale centésimos de bit, não décimos.


## 13. Contra os k-quants do llama.cpp (Q3_K_M a Q8_0)

Medido em 15 e 16 de outubro de 2026 em `unsloth/Llama-3.2-1B`, wikitext-2 inteiro (141 janelas de 2048, 288.627 tokens), contas em float32. Os GGUFs saíram do `llama-quantize` (llama.cpp `2f729cf`) a partir do GGUF BF16 do mesmo safetensors, **sem imatrix**. Para cada GGUF, o `gguf-py` desquantiza as mesmas 112 matrizes que o STRATA corta e o `strata_ppl.py --gguf` as coloca no modelo; embedding e normas ficam em BF16 nas duas comparações. Os bits/peso do GGUF são os bytes reais dessas matrizes dentro do arquivo. Conferência: com o GGUF BF16, as 112 matrizes voltam bit a bit idênticas ao safetensors.

A rodada do llama.cpp foi numa CPU de 4 núcleos (PPL BF16 9,7506) e a do STRATA numa T4 do Colab (9,7509). A diferença entre as duas referências é de 0,003%, cem vezes menor que qualquer ΔPPL da tabela; cada ΔPPL é contra a referência da própria máquina. Números em [`tests/ppl/resultados/unsloth__Llama-3.2-1B_gguf.json`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_gguf.json) (Q4_K_M, Q5_K_M, Q6_K) e [`_gguf2.json`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_gguf2.json) (Q3_K_M, Q8_0; outra rodada na mesma CPU, referência 9,7506 de novo).

| Formato | Bits/peso no disco | ΔPPL % | Bits/peso na RAM |
| --- | ---: | ---: | ---: |
| STRATA E0 (0.2, rANS) | 3,608 | +155,08% | 16 |
| **llama.cpp Q3_K_M** | **3,842** | **+20,38%** | 3,8 |
| STRATA E1 (0.2, rANS) | 4,594 | +13,08% | 16 |
| **llama.cpp Q4_K_M** | **4,802** | **+4,07%** | 4,8 |
| INT8 progressivo, 5 bits (com entropia) | 5,067 | +5,31% | 5,5 |
| STRATA E2 | 5,594 | +2,74% | 16 |
| **llama.cpp Q5_K_M** | **5,656** | **+1,25%** | 5,7 |
| INT8 progressivo, 6 bits (com entropia) | 6,047 | +1,14% | 6,5 |
| **llama.cpp Q6_K** | **6,562** | **+0,49%** | 6,6 |
| STRATA E3 | 6,594 | +0,61% | 16 |
| INT8 progressivo, 7 bits (com entropia) | 7,023 | +0,51% | 7,5 |
| STRATA E4 | 7,594 | +0,15% | 16 |
| **llama.cpp Q8_0** | **8,500** | **+0,03%** | 8,5 |
| STRATA E5 | 8,594 | +0,06% | 16 |

Q4_K_M e Q5_K_M usam Q6_K em 14,7% dos pesos (`attn_v` e parte de `ffn_down`) e o tipo base no resto.

**O STRATA perde em todas as faixas medidas.**

- **Abaixo de 4,6 bits o STRATA não tem modelo utilizável:** o E0 (só sinal e expoente) degrada 155%. O Q3_K_M entrega +20,38% com 3,84 bits; o STRATA só passa disso no E1, com 0,75 bit a mais.
- **Perto de 4,8 bits:** o Q4_K_M degrada 3,2 vezes menos que o E1 (4,07% contra 13,08%) com 0,21 bit a mais. Para chegar à qualidade do Q4_K_M o STRATA precisa ler o E2, 5,59 bits: 16% mais download, e ainda com 16 bits na RAM.
- **Perto de 5,6 bits:** o Q5_K_M degrada 2,2 vezes menos que o E2 (1,25% contra 2,74%) com 0,06 bit a mais. Igualar exige o E3, 6,59 bits (+0,94).
- **Perto de 6,6 bits:** o Q6_K ganha por pouco do E3 (0,49% contra 0,61%) com 0,03 bit a menos. É onde a diferença é menor.
- **Perto de 8,5 bits:** o Q8_0 (+0,03%) ganha do E5 (+0,06%) com 0,09 bit a menos. Nessa faixa as duas diferenças já ficam abaixo do que qualquer avaliação de tarefa enxerga.
- Interpolando a curva do STRATA em log(ΔPPL), ele precisaria de 0,54 bit a mais que o Q4_K_M, 0,46 a mais que o Q5_K_M e 0,19 a mais que o Q6_K para a mesma qualidade. Mas o STRATA não tem meio estrato: na prática paga o estrato inteiro seguinte.
- **A comparação favorece o STRATA**, e o placar real é pior: (1) os GGUFs estão sem imatrix, que é como quase todo GGUF publicado é feito e melhora justamente o Q4/Q5; (2) os GGUFs estão no tamanho nominal, sem codificador de entropia, enquanto o STRATA tem rANS. A entropia de ordem 0 dos índices de 4 bits nas matrizes Q4_K é 3,86 bits, então um GGUF com o mesmo tipo de codificador cairia uns 0,12 bit/peso; (3) na RAM o Q4_K_M ocupa 4,8 bits e roda em kernel próprio, e o corte do STRATA volta a 16.
- **Contra o INT8 progressivo** (a outra forma de ter um arquivo que se refina), com codificador de entropia dos dois lados, é quase empate: o progressivo erra um pouco menos em 5 e 6 bits (5,31% contra cerca de 6,2%; 1,14% contra cerca de 1,4%, interpolando o STRATA) e um pouco mais em 7 bits (0,51% contra cerca de 0,33%). Os dois perdem para os k-quants.

**Por quê.** Um estrato do STRATA carrega sinal e expoente inteiros em cada peso: 3,6 bits antes do primeiro bit de mantissa. O k-quant paga a magnitude uma vez por bloco (escala e mínimo de 6 bits a cada 32 pesos, dentro de super-blocos de 256) e gasta todos os outros bits em níveis uniformes dentro do bloco, escolhidos para minimizar o erro. Truncar um float é um quantizador logarítmico por peso; para pesos com essa distribuição, níveis uniformes por bloco gastam menos bits pelo mesmo erro. Melhorar o codificador do expoente não fecha isso: depois do rANS sobram uns 0,07 bit (seção 12), e a distância para o Q4_K_M é de 0,5 a 0,8.

**O que sobra para o STRATA.** Como formato de uso com perdas, nenhum estrato é a melhor escolha em sua faixa de tamanho. O que continua de pé é o arquivo sem perdas (10,59 b/peso, 66% do BF16, na faixa do DFloat11), a genealogia (−19% para um fine-tune dado o pai) e a leitura progressiva como recurso secundário, não como substituto dos GGUFs. O caminho que a especificação já propõe para competir com perdas é o modo residual ([B.1](02-especificacao-formato.md#b1-modo-residual)): primeiro estrato = um quantizador de verdade (por exemplo os blocos do Q4_K), depois resíduos até o BF16 exato. Ainda não foi medido quanto isso custa no tamanho sem perdas.

## 14. Estrato por tensor

Pergunta: o Q4_K_M ganha em parte por misturar tipos por camada. Se cada uma das 112 matrizes receber o seu próprio estrato, escolhido pela sensibilidade, o STRATA fecha a diferença com os mesmos bits? Rodado com [`tests/ppl/strata_alloc.py`](../tests/ppl/strata_alloc.py) numa T4, wikitext-2 de teste inteiro (288 627 tokens). Sensibilidade = proxy de Fisher diagonal (Σ g²·Δw²) em 32 752 tokens do wikitext-2 de **treino**. Alocação por mochila de múltipla escolha nos bits/peso medidos de cada k-quant. Tabela em [`resultados/unsloth__Llama-3.2-1B_alloc.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_alloc.md).

| | bits/peso | ΔPPL | k-quant com os mesmos bits |
|---|---:|---:|---|
| E1 uniforme | 4,59 | +13,08% | |
| **por tensor** | **4,80** | **+8,91%** | Q4_K_M +4,07% |
| E2 uniforme | 5,59 | +2,74% | |
| **por tensor** | **5,66** | **+2,30%** | Q5_K_M +1,25% |
| **por tensor** | **6,56** | **+0,59%** | Q6_K +0,49% |
| E3 uniforme | 6,59 | +0,61% | |

**Ajuda pouco e não muda o placar.** O STRATA continua perdendo nos três tamanhos.

- O proxy funciona: a razão ΔPPL/proxy fica entre 4 300 e 5 700 nas seis linhas, e o ajuste em log-log tem expoente 1,09. A alocação não é ruim por causa do proxy. Medir a sensibilidade real por matriz (mais caro) não deve mudar a conclusão.
- Pelo proxy calibrado, a alocação por tensor precisaria de 5,31, 6,08 e 6,67 bits para igualar Q4_K_M, Q5_K_M e Q6_K: **+0,51, +0,42 e +0,11 bit**. Com estrato uniforme interpolado eram +0,54, +0,46 e +0,19. A alocação recupera de 0,03 a 0,08 bit numa distância de 0,1 a 0,5.
- Motivo: a sensibilidade por peso varia só 31 vezes entre as 112 matrizes (coeficiente de variação 0,94), e não há onde economizar a parte cara. Os 3,6 bits de sinal e expoente são pagos em todo peso, em qualquer estrato. A alocação mexe só nos planos de mantissa. Os estratos escolhidos sobem k e v (2,4 a 4,3) e baixam gate/up (1,0 a 3,0), como a mistura do Q4_K_M, mas sem mexer no piso.
- Formato: estrato por tensor não pede mudança no arquivo. É só a ordem de leitura dos planos (um mapa de 112 bytes), então pode virar o perfil padrão de leitura parcial sem custo.

Conclusão: a distância para os k-quants é estrutural, como na seção 13. O próximo teste que pode mudar o placar é o modo residual ([B.1](02-especificacao-formato.md#b1-modo-residual)).

## 15. Modo residual: quanto custa o arquivo sem perdas

Pergunta: se o primeiro estrato for um k-quant de verdade (spec [B.1](02-especificacao-formato.md#b1-modo-residual)) e o resto fechar o BF16 exato, quanto o arquivo sem perdas cresce? Medido na CPU com [`tests/ppl/strata_residual.py`](../tests/ppl/strata_residual.py), sem rodar o modelo. Tabela em [`resultados/unsloth__Llama-3.2-1B_residual.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_residual.md).

Como: o k-quant arredonda cada peso ao nível mais próximo da sua grade, então quem leu o estrato 0 sabe que o peso está numa célula de largura `d` em volta do valor desquantizado. O resíduo só escolhe o BF16 dentro da célula, ao custo de `-log2 p(w) + log2 P(célula)`, com `p` = tabela de sinal+expoente por tensor (a mesma do rANS do core, por oitava do passo) e mantissa uniforme. Nenhum dos 973 M pesos caiu fora da célula, o que confere a grade.

| Estrato 0 | bits do estrato 0 | resíduo | **sem perdas** | vs STRATA 0.2 |
|---|---:|---:|---:|---:|
| sinal+expoente (STRATA 0.2) | 3,60 | 6,99 | **10,59** | |
| Q4_K_M | 4,80 | 6,43 | **11,24** | +0,64 |
| Q5_K_M | 5,66 | 5,57 | **11,23** | +0,63 |
| Q6_K | 6,56 | 4,64 | **11,20** | +0,61 |

- **O custo é fixo, cerca de 0,6 bit/peso (6%), qualquer que seja a base.** O resíduo recupera quase todos os bits dos índices do k-quant. O que não volta são as escalas e mínimos por bloco (0,5 bit/peso no Q4_K/Q5_K, 0,56 no Q6_K), que o arquivo sem perdas puro não precisa. Um modelo a priori melhor (histograma inteiro por tensor, sem contar o cabeçalho) só desce mais 0,02 a 0,05.
- **Armazenamento.** Um arquivo de 11,24 bits/peso cujo começo é o Q4_K_M byte a byte e cujo fim é o BF16 exato substitui o par "BF16 + Q4_K_M": 15,39 bits/peso com o BF16 comprimido (−27%) ou 20,8 com o safetensors cru (−46%).
- **O que falta medir:** se os níveis intermediários entre o Q4_K_M e o BF16 servem de Q5/Q6. Dividir a célula ao meio custa perto de 1 bit e dá uma grade Q5-like com as escalas do Q4_K; se o ΔPPL a ~5,8 e ~6,8 bits ficar perto do Q5_K_M e do Q6_K, um arquivo substitui quatro.

## 16. Modo residual: os níveis intermediários

Pergunta: no arquivo da seção 15, os níveis entre o Q4_K_M e o BF16 servem de Q5 e Q6? Nível k = célula do Q4_K dividida em 2^k partes iguais, valor lido no centro da sub-célula; pesos que saturam nas pontas da grade ficam exatos já no nível 1. ΔPPL numa T4 ([`strata_residual_colab.ipynb`](../tests/ppl/strata_residual_colab.ipynb)), wikitext-2 de teste inteiro, com Q5_K_M e Q6_K na mesma rodada. Tamanhos medidos na CPU (seção 15). Tabelas em [`resultados/unsloth__Llama-3.2-1B_residual_ppl.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_residual_ppl.md) e [`_residual.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_residual.md).

| Leitura do mesmo arquivo | bits/peso | ΔPPL | k-quant separado |
|---|---:|---:|---|
| nível 0 = Q4_K_M byte a byte | 4,80 | +4,07% | Q4_K_M 4,80 b +4,07% |
| nível 1 | 5,95 | **+1,17%** | Q5_K_M 5,66 b +1,25% |
| nível 2 | 6,95 | **+0,31%** | Q6_K 6,56 b +0,49% |
| nível 3 | 7,94 | +0,11% | Q8_0 8,50 b +0,03% |
| tudo | 11,24 | 0 (BF16 exato) | |

- **Cada nível intermediário erra menos que o k-quant da sua faixa**, e custa um pouco mais. Interpolando em log(ΔPPL), a qualidade do Q5_K_M sai com 5,89 bits (+0,23) e a do Q6_K com 6,60 (+0,04, praticamente empate). Pela primeira vez uma leitura parcial do STRATA está na fronteira dos k-quants.
- O nível 1 custa 1,15 bit em vez de 1,0 porque guarda exatos os pesos que saturam nas pontas da grade (0,14% dos pesos). O ganho de qualidade sobre o Q5_K_M vem em parte disso.
- **Armazenamento.** O arquivo de 11,24 bits/peso substitui BF16 + Q4_K_M + Q5_K_M + Q6_K. Guardados separados, eles somam 27,6 bits/peso com o BF16 comprimido (−59%) ou 33,0 com o safetensors cru (−66%). Somando o Q8_0 aos crus, 41,5 (−73%).
- Com a lista de leitura por tensor (B.7), o leitor pode parar entre dois níveis, então os tamanhos intermediários (5,7, 6,6...) também existem.
- **O que ainda falta para rodar:** os níveis 1 a 3 não são um tipo GGUF existente. O nível 0 roda direto no llama.cpp. Os outros precisam de um kernel "Q4_K + bits de refinamento" ou de uma conversão para Q5_K/Q6_K na hora de carregar. A grade do nível k tem passo d/2^k com as mesmas escalas, então a conversão pode ser quase exata; isso ainda não foi medido.

## 17. Outros tipos de modelo: voz e difusão (FP16 e FP32)

Pergunta: o formato sem perdas funciona fora dos modelos de texto em BF16? Medido na CPU com [`tests/test-tamanho.mjs`](../tests/test-tamanho.mjs), que codifica e decodifica tensor por tensor e confere o SHA-256 do arquivo remontado. Números em [`resultados/tamanho__whisper_sd-turbo.json`](../tests/resultados/tamanho__whisper_sd-turbo.json).

| Modelo (arquivo inteiro) | tipo | pesos | original | .strata | fração | bits/peso | estrato 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B (seção 12, referência) | BF16 | 1,24 G | | | 66,2% | 10,59 de 16 | 3,61 |
| openai/whisper-large-v3-turbo (voz → texto) | FP16 | 809 M | 1617,8 MB | 1356,3 MB | **83,8%** | 13,41 de 16 | 3,48 |
| stabilityai/sd-turbo, UNet (difusão) | FP16 | 866 M | 1731,8 MB | 1468,0 MB | **84,8%** | 13,56 de 16 | 3,69 |
| openai/whisper-small (voz → texto) | FP32 | 242 M | 966,9 MB | 410,0 MB | 42,4% | 13,57 de 32 | 3,59 |

- **A volta é bit a bit nos três** (SHA-256 igual; 1.755 conferências, 0 falhas). O formato não depende do que o modelo faz: só vê tensores de ponto flutuante.
- **Quem decide o ganho é o tipo numérico, não a modalidade.** Sinal + expoente custam cerca de 3,5 bits em qualquer um dos modelos (3,48 a 3,69); o que muda é quantos bits de mantissa sobram, e esses são quase aleatórios. BF16 tem 7 bits de mantissa (economia de ~5,4 bits de 16, −34%); FP16 tem 10 (economia de ~2,5 bits, −16%).
- **O FP32 do whisper-small é FP16 disfarçado.** Os 13 bits finais da mantissa são zero em 100% dos pesos; os 42,4% vêm de descartar esses zeros, que qualquer compressor acha, e o custo real (13,57 b) é o mesmo do FP16. Não é mérito do STRATA.
- **Consequência para a meta de armazenamento:** os modelos grandes de texto, que dominam o volume dos hubs, saem em BF16 e são o caso forte (−34%). Modelos de voz e difusão em FP16 ganham metade disso sem perdas; para eles o valor está mais nos estratos parciais do que no arquivo sem perdas. A qualidade dos estratos parciais nesses modelos ainda não foi medida (precisa de WER para voz e FID/CLIP para imagem, não ΔPPL).
- **O que falta para separar tipo de modalidade:** um modelo de imagem ou voz salvo em BF16. Se ele ficar perto de 66%, a regra "o tipo decide" está confirmada.
