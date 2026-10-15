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

