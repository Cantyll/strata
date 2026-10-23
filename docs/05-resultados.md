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

Pergunta: o formato sem perdas funciona fora dos modelos de texto em BF16? O codificador de imagem do SmolVLM foi recortado do arquivo (tensores `vision_model`) para medir uma parte não textual salva em BF16. Medido na CPU com [`tests/test-tamanho.mjs`](../tests/test-tamanho.mjs), que codifica e decodifica tensor por tensor e confere o SHA-256 do arquivo remontado. Números em [`resultados/tamanho__whisper_sd-turbo.json`](../tests/resultados/tamanho__whisper_sd-turbo.json).

| Modelo (arquivo inteiro) | tipo | pesos | original | .strata | fração | bits/peso | estrato 0 |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B (seção 12, referência) | BF16 | 1,24 G | | | 66,2% | 10,59 de 16 | 3,61 |
| openai/whisper-large-v3-turbo (voz → texto) | FP16 | 809 M | 1617,8 MB | 1356,3 MB | **83,8%** | 13,41 de 16 | 3,48 |
| stabilityai/sd-turbo, UNet (difusão) | FP16 | 866 M | 1731,8 MB | 1468,0 MB | **84,8%** | 13,56 de 16 | 3,69 |
| openai/whisper-small (voz → texto) | FP32 | 242 M | 966,9 MB | 410,0 MB | 42,4% | 13,57 de 32 | 3,59 |
| SmolVLM-256M, só o codificador de imagem | BF16 | 86 M | 172,9 MB | 114,4 MB | **66,2%** | 10,59 de 16 | 3,60 |

- **A volta é bit a bit nos quatro** (SHA-256 igual; 1.953 conferências, 0 falhas). O formato não depende do que o modelo faz: só vê tensores de ponto flutuante.
- **Quem decide o ganho é o tipo numérico, não a modalidade.** Sinal + expoente custam cerca de 3,5 bits em qualquer um dos modelos (3,48 a 3,69); o que muda é quantos bits de mantissa sobram, e esses são quase aleatórios. BF16 tem 7 bits de mantissa (economia de ~5,4 bits de 16, −34%); FP16 tem 10 (economia de ~2,5 bits, −16%).
- **O FP32 do whisper-small é FP16 disfarçado.** Os 13 bits finais da mantissa são zero em 100% dos pesos; os 42,4% vêm de descartar esses zeros, que qualquer compressor acha, e o custo real (13,57 b) é o mesmo do FP16. Não é mérito do STRATA.
- **Consequência para a meta de armazenamento:** os modelos grandes de texto, que dominam o volume dos hubs, saem em BF16 e são o caso forte (−34%). Modelos de voz e difusão em FP16 ganham metade disso sem perdas; para eles o valor está mais nos estratos parciais do que no arquivo sem perdas. A qualidade dos estratos parciais nesses modelos ainda não foi medida (precisa de WER para voz e FID/CLIP para imagem, não ΔPPL).
- **Tipo separado de modalidade:** o codificador de imagem em BF16 dá 66,2% e 3,60 bits no estrato 0, os mesmos números do Llama em BF16. A regra "o tipo decide" vale para o que foi medido (texto, imagem, voz, difusão; até 1,2 G pesos).

## 18. Segurança do leitor

Pergunta: o `.strata` mantém a segurança do `.safetensors`? O safetensors é seguro porque não tem código executável e porque o leitor confere todo offset e tamanho antes de usar; não há nada para descomprimir. O `.strata` também não tem código executável, mas descomprime (rANS e deflate) e confia em offsets, tamanhos e tipos escritos no cabeçalho. Isso é superfície de ataque que o safetensors não tem. Medido com [`tests/fuzz-leitor.mjs`](../tests/fuzz-leitor.mjs): arquivos válidos pequenos (o exemplo da spec, um modelo sintético e um de tipos mistos) mutados de 15 jeitos (bits trocados, corte, campos com valores absurdos, streams repetidos ou trocados, bombas de descompressão, fonte adulterada, hash refeito por quem escreve o arquivo), cada caso levado até a volta para `.safetensors`.

| Classe | O que significa | Leitor antes | Endurecido, sem nenhum hash | Endurecido, com hashes |
|---|---|---:|---:|---:|
| recusa | recusou com um erro dele | 28,9% | 91,0% | 98,4% |
| intacto | leu e devolveu o original exato (a mutação não mudou nada) | 22,7% | 1,6% | 1,6% |
| **silencioso** | **leu sem erro e devolveu pesos diferentes** | **18,9%** | 7,4% | **0** |
| quebra | `TypeError`/`RangeError`: usou um campo sem conferir | 22,9% | 0 | 0 |
| recurso | tentou alocar ou descomprimir mais de 64 MB | 6,7% | 0 | 0 |
| pior caso | | 544 ms | 24 ms | 19 ms |

30.000 casos por coluna, semente 11. Leitor endurecido em JS; o leitor em Rust deu exatamente as mesmas classes.

- **Antes**, 2.001 casos tentavam alocar gigabytes: um stream rANS de um símbolo só custa zero bit por elemento, então 9 bytes declaravam 2^31 expoentes; um deflate de 128 KB virava 128 MB sem limite. 6.860 casos estouravam erros de JavaScript porque um offset negativo, um `t` fora da tabela ou uma parte `__proto__` era usado sem conferir. E 22,7% dos arquivos malformados (dados sobrando no fim, stream repetido) eram aceitos.
- **As regras de A.12** (validação completa do cabeçalho antes de tocar no corpo, saída limitada ao declarado, inflate próprio que para no tamanho certo) zeram quebra e recurso, e o pior caso cai de 544 para 24 ms.
- **O 7,4% silencioso que sobra sem hash nenhum é irredutível**: são bits trocados dentro de planos crus ou de um deflate que continua válido. Nenhum leitor de nenhum formato pega isso sem hash; o safetensors aceitaria do mesmo jeito.
- **Com os hashes de A.13, silencioso vai a zero.** O hash por stream pega o byte trocado antes de chegar ao descompressor; quando quem escreve o arquivo refaz esse hash, o do tensor pega na saída.

**Dois leitores, os mesmos casos.** O leitor de referência em Rust ([`leitor-rs/`](../leitor-rs)) foi escrito a partir da especificação, com outro inflate (miniz_oxide) e outro parser de JSON (serde_json). Nos 90.000 casos acima (as três colunas), ele aceitou e recusou exatamente os mesmos arquivos que o JS, e devolveu os mesmos bytes nos aceitos, inclusive nos 2.231 silenciosos sem hash, em que os dois devolvem o mesmo lixo. Isso confere que as regras de A.12 não deixam ambiguidade e que o inflate escrito à mão no JS se comporta como o miniz. [`tests/vetores/`](../tests/vetores) guarda 22 desses casos (3 válidos, 19 hostis) com o resultado esperado; os dois leitores passam em todos.

**Num modelo real (Llama-3.2-1B, 146 tensores, 1.314 streams).** Medido com [`tests/medir-leitores.mjs`](../tests/medir-leitores.mjs) e `strata-ref confere`, num contêiner de 4 núcleos:

| | Resultado |
|---|---|
| Custo dos hashes | cabeçalho de 132 KB → 237 KB: +104 KB, 0,0064% do arquivo, 0,0007 bit/peso |
| `fonte.sha256` | `68a2e4be…6809a`, igual ao SHA-256 que o Hugging Face publica para o `model.safetensors` |
| Volta em JS, leitor antigo | 156 s |
| Volta em JS, endurecido, com todos os hashes | 152 s, SHA-256 idêntico |
| Decodificação em Rust, com todos os hashes | 31 a 38 s (65 a 80 MB/s de saída), SHA-256 da volta idêntico |

- **A segurança não custou tempo no JS**: o inflate próprio é mais rápido que o `DecompressionStream` do Node (68 contra 38 MB/s de saída), o que paga os hashes.
- O Rust é 4 a 5 vezes mais rápido. (Uma versão anterior deste texto dizia que cerca de 10 s eram SHA-256 em software porque o crate `sha2` não detectava a instrução SHA desta CPU. Estava errado: a detecção funciona e o hash sai a cerca de 1,5 GB/s; ver "SHA-256 e memória desta máquina" abaixo.)

**Escolha entre os dois caminhos.** Segurança deu empate: as mesmas recusas, caso a caso. Quem fez o trabalho foram as regras de A.12 e os hashes de A.13, não a linguagem. A diferença está em velocidade (Rust 4 a 5 vezes mais rápido) e em onde cada um roda: o JS é o único que roda na página, no navegador; o Rust é a base natural para um leitor em Python (como o próprio safetensors, que é Rust por baixo) e para carregadores como vLLM e llama.cpp. Por isso o Rust fica como leitor de referência e base dos carregadores, e o JS continua só na página, preso aos mesmos vetores e ao mesmo fuzzing diferencial.

**Leitor em Python, por cima do Rust.** [`leitor-py/`](../leitor-py) expõe o leitor de referência ao Python com a mesma API do safetensors (`load_file`, `safe_open`, `keys`, `get_tensor`) e devolve arrays do numpy ou tensores do torch sem copiar: cada tensor é um buffer com dono em Rust que os dois leem direto. Toda a validação e os hashes são os do Rust. Para isso o leitor ganhou duas coisas: decodificação em várias threads, dividindo também os tensores grandes (o embedding do Llama-3.2-1B é um quinto do modelo e, sozinho numa thread, levava 24 s), e `bytes_ate(prof)`, que diz pelo cabeçalho quantos bytes baixar para ler numa profundidade. Medido no mesmo contêiner de 4 núcleos, que oscila muito (a mesma decodificação numa thread levou de 27 a 60 s ao longo do dia), por isso as comparações abaixo são de rodadas alternadas:

| Llama-3.2-1B | Resultado |
|---|---|
| `strata-ref confere`, 1 thread contra 4 | 52 a 60 s contra 16 a 19 s (cerca de 3,3 vezes) |
| `strata_ref.load_file(..., framework="pt")`, todos os planos | 35 a 36 s, pico de 4,4 GB de RAM; os 146 tensores idênticos, bit a bit, aos do `safetensors.torch.load_file` no original |
| Prefixo de 3 planos (1,02 GB, 6,59 bits/peso) | 30 s |
| Prefixo só de sinal e expoente (556 MB, 3,60 bits/peso) | 17 s |

- **Ler um prefixo não é muito mais rápido que ler tudo.** O que custa é, nesta máquina virtual, sobretudo escrever a saída em memória nova (o tempo de sistema passa do de usuário), mais o rANS do expoente e o SHA-256. Esses custos são iguais em qualquer profundidade.
- **Ler de um disco local, o `.strata` sempre perde.** O safetensors mapeia o arquivo e devolve os tensores quase de graça. O `.strata` precisa decodificar.
- **O `.strata` só ganha onde o gargalo é a rede.** Baixar 1,64 GB e decodificar em 36 s fica mais rápido que baixar 2,47 GB de safetensors quando a conexão está abaixo de cerca de 23 MB/s (≈ 190 Mbit/s) nesta máquina de 4 núcleos. Acima disso, o safetensors chega antes. Por isso o uso certo é converter uma vez (`strata-ref volta`, ou `para_safetensors`) e carregar o arquivo convertido nas vezes seguintes.

**Decodificar enquanto baixa.** O leitor ganhou um modo em fluxo (`Fluxo` no Rust, `load_stream` e `load_url` no Python): cada stream é conferido pelo hash e descomprimido assim que termina de chegar, cada tensor é montado (e seu hash conferido) assim que chega o último plano dele, e os bytes já usados saem da memória. Com `prof`, o download para no último byte daquele estrato. Ele aceita e recusa o mesmo que a leitura normal com a mesma profundidade: nos 22 vetores, em pedaços de 1 byte até o arquivo inteiro e em todas as profundidades, e em 4.500 arquivos mutados (`leitor-rs/tests/fluxo.rs`).

Medido no Llama-3.2-1B com `strata-ref fluxo arquivo 8 taxa` (pedaços de 8 MB entregues no ritmo da taxa), alternando a primeira versão do fluxo (que juntava todos os tensores no fim) e a atual, duas rodadas de cada, contra baixar o safetensors (2,47 GB) e contra baixar o `.strata` e só depois decodificar (36 s):

| Rede | safetensors | baixa, depois decodifica | fluxo, 1ª versão | fluxo, atual | sobra depois do último byte (1ª → atual) |
|---|---:|---:|---:|---:|---:|
| 100 Mbit/s (12,5 MB/s) | 198 s | 167 s | 135 s | **131 s** | 4 s → 0 |
| 200 Mbit/s (25 MB/s) | 99 s | 101 s | 70 a 75 s | **68 a 77 s** | 4 a 10 s → 0 |
| 320 Mbit/s (40 MB/s) | 62 s | 77 s | 44 a 48 s | **42 a 56 s** | 3 a 7 s → 0 |
| 500 Mbit/s (62,5 MB/s) | 40 s | 62 s | 30 a 35 s | **29 a 45 s** | 4 a 9 s → 0 |

- **A sobra depois do último byte foi a zero.** Montar cada tensor quando chega o último plano dele espalha a junção pelo último estrato do download, em vez de deixá-la toda para o fim.
- **O tempo total quase não mudou.** O trabalho saiu do fim, mas continua lá: nas taxas mais altas o decodificador fica um pouco atrás da rede no último estrato, que é quando todos os tensores ficam completos. As diferenças entre as duas versões estão dentro da oscilação da máquina.
- **Com mais rodadas, o empate com o safetensors subiu para entre 320 e 500 Mbit/s.** A estimativa anterior, de cerca de 250 Mbit/s, vinha de uma rodada só por taxa medida pelo Python, numa hora em que a máquina estava mais lenta. Nas rodadas alternadas, as duas versões do fluxo chegam antes do safetensors a 320 Mbit/s e empatam a 500.
- **Antes, uma tentativa pior: fundir cada plano no peso assim que ele chega.** A sobra caiu para 1 s, mas cada plano passava pela saída inteira (9 passadas de leitura e escrita em 2,5 GB, contra 1 da junção), e o tempo de CPU ficou 2,5 vezes maior. Foi descartada.
- **O pico de RAM caiu de 4,4 GB (leitura normal) para 3,3 GB.**
- **A máquina oscila muito**: a mesma rodada a 500 Mbit/s levou de 29 a 45 s. Por isso as faixas, e por isso as duas versões foram alternadas.

**SHA-256 e memória desta máquina.** O `sha2` 0.10 já usa a instrução SHA da CPU (SHA-NI), detectada em tempo de execução: em memória já tocada, o hash sai a cerca de 1,5 GB/s, o que dá menos de 2 s para os 2,47 GB do Llama-3.2-1B. Não há nada a ligar no leitor. O que inflava os tempos é outra coisa: neste contêiner, a primeira escrita em memória recém-alocada é muito lenta (alocar e preencher 1 GiB leva cerca de 16 s; tocar um byte por página do mesmo GiB de novo, 4,8 s). Por isso o tempo de sistema passa do de usuário em todas as medições acima, e por isso elas oscilam tanto. Numa máquina comum esse custo é bem menor, então os tempos de decodificação daqui são pessimistas.

**Expoente em blocos independentes (B.5), medido sem mudar o formato.** Hoje o expoente de cada tensor é um stream rANS só, com um estado: uma thread decodifica do começo ao fim. No embedding (262,7 milhões de símbolos) isso leva 1,65 s com qualquer número de threads, e ele é o maior trabalho serial do leitor. Dividir esse stream em blocos que começam com estado próprio, todos com a tabela de frequências do tensor e um offset de 4 bytes por bloco, permite decodificar os blocos em paralelo (em núcleos hoje, em GPU no kernel). Medido com [`leitor-rs/examples/rans_blocos.rs`](../leitor-rs/examples/rans_blocos.rs), que reproduz o codificador em JS byte a byte (o bloco único dá exatamente os 401.777.977 B dos streams de expoente do `.strata`):

| Tamanho do bloco | Blocos no modelo | Bytes a mais | bit/peso a mais |
|---|---:|---:|---:|
| 1 M símbolos | 1.212 | +8.323 | 0,00005 |
| 256 K | 4.747 | +34.928 | 0,00023 |
| **64 K** | **18.889** | **+140.963** | **0,00091** |
| 16 K | 75.457 | +565.248 | 0,00366 |
| 4 K | 301.729 | +2.262.329 | 0,01465 |

| Embedding, decodificação do expoente | 1 thread | 4 threads |
|---|---:|---:|
| Um bloco (hoje) | 1,65 s | 1,65 s |
| Blocos de 1 M, 64 K ou 4 K | cerca de 1,58 s | cerca de 0,43 s (3,9 vezes) |

- **O custo é desprezível até 64 K**: 141 KB num arquivo de 1,64 GB. Cada bloco paga 4 bytes de offset e os poucos bytes de estado final; a tabela continua uma por tensor.
- **Em CPU, qualquer tamanho até 1 M já dá a aceleração toda** com 4 núcleos. Blocos menores só servem para ter trabalho suficiente para milhares de threads, que é o caso da GPU.
- **64 K é a recomendação**: perto do que o DFloat11 usa, ainda grátis em tamanho, e com cerca de 19 mil blocos no modelo de 1 B (mais de 1 milhão num de 70 B), o bastante para ocupar uma GPU.
- **Isto ainda não está no formato.** Gravar em blocos muda o arquivo (uma versão nova ou um campo no stream de expoente, o codificador em JS e os dois leitores), então ficou como medição.

**Comparado ao safetensors:**

| Propriedade | safetensors | `.strata` antes | `.strata` agora |
|---|---|---|---|
| Código executável no arquivo | não | não | não |
| Teto do cabeçalho | 100 MB | nenhum (4 GB) | 100 MB |
| Offsets contíguos, sem sobreposição, dentro do arquivo | conferidos | não conferidos | conferidos |
| Tamanho = shape × dtype | conferido | não | conferido |
| Descompressão | não existe | sem limite | para no tamanho declarado |
| Saída máxima | o próprio arquivo | ilimitada | o declarado, até `max(1 GiB, 64 × arquivo)`, ajustável |
| Hash dentro do arquivo | nenhum (o Git LFS confere o arquivo inteiro) | nenhum | por stream, por tensor e do original |
| Leitura parcial conferida | não se aplica | não | sim, por stream |
| Leitores independentes | 1 (Rust, com bindings) | 1 (JS) | 2 (JS e Rust), com fuzzing diferencial |

**O que isto não prova.**

- O fuzzer é de mutação, escrito aqui: 90 mil casos na rodada final e cerca de 200 mil durante o desenvolvimento. Não é guiado por cobertura (libFuzzer, AFL), que acha caminhos que mutação cega não acha. O contêiner não tem o Rust nightly que o `cargo fuzz` exige.
- Descomprimir continua sendo superfície que o safetensors não tem. O que a reduz: as duas implementações são em linguagens com memória segura, a saída é limitada antes de alocar, e com um id publicado nenhum byte não conferido chega ao descompressor.
- Hashes dentro do arquivo não protegem contra quem escreveu o arquivo. Isso depende de uma âncora de fora: o id publicado (pelo índice multi-arquivo ou pelo hub), ou o `fonte.sha256` comparado com o SHA-256 que o Hugging Face já publica, ou uma assinatura (ainda não implementada).
- No navegador, tensores acima de 256 MB são conferidos pelo SHA-256 em JS (cerca de 120 MB/s), porque o SubtleCrypto do Node recusa entradas desse tamanho e o limite dos navegadores não foi medido.

## 19. Modelos em vários arquivos (Llama-3.1-8B, 4 shards)

Pergunta: um modelo que chega em vários `.safetensors` vira um `.strata` só, um por shard, ou outra divisão? Dois desenhos foram implementados e medidos sobre **os mesmos streams** do core (spec [B.9](02-especificacao-formato.md#b9-modelos-em-vários-arquivos)):

- **D1, por shard:** um `.strata` 0.2 por shard original (byte a byte o que o conversor de um arquivo grava) e um índice com o prefixo de cada estrato.
- **D2, por estrato:** um arquivo por estrato com todos os tensores do modelo, em partes de até 5 GB, e um índice com a tabela de streams e os cabeçalhos originais.

Medido em 17 de outubro de 2026 em `unsloth/Meta-Llama-3.1-8B` (cópia aberta do Llama-3.1-8B: 8.030.261.248 pesos BF16, 4 shards, 16,06 GB), num contêiner de 4 núcleos e 15 GB de RAM, Node 22. Esta rodada usou o core antes dos hashes da A.13 (sem `h` por stream e `sha256` por tensor); os hashes estão ligados nos dois desenhos e testados no modelo sintético de `node tests/test-shards.mjs`. JSON: [`tests/resultados/shards__unsloth__Meta-Llama-3.1-8B.json`](../tests/resultados/shards__unsloth__Meta-Llama-3.1-8B.json).

```
node tests/test-shards.mjs encode <pasta-do-modelo> <trabalho> model-00001-of-00004.safetensors   # um processo por shard
node tests/test-shards.mjs d1 <pasta-do-modelo> <trabalho> <saída-d1>
node tests/test-shards.mjs d2 <pasta-do-modelo> <trabalho> <saída-d2> --cap-gb 5
node tests/test-shards.mjs verify <saída> --depth 2 --json r.json
```

**Comum aos dois** (a codificação é a mesma):

| | |
|---|---|
| Sem perdas | 10.608.264.128 bytes de corpo, **66,1%** do original, 10,57 bits/peso (E0 3,58 bits; cada plano da mantissa ≈ 1,0) |
| Codificar | 2.466 s de CPU somando os 4 shards (3,3 M pesos/s por núcleo); 3 shards em paralelo, ≈ 13 min de relógio |
| Memória | 4,6 GB no shard com o `lm_head` (525 M pesos num tensor só); 1,3 GB nos outros |
| Volta | os 4 shards com o SHA-256 que o Hugging Face publica para cada um, e o `model.safetensors.index.json` idêntico, nos dois desenhos |

**Onde diferem:**

| | D1 por shard | D2 por estrato |
|---|---:|---:|
| Arquivos | 4 (maior: 3,30 GB) | 8 (maior: 3,60 GB, o estrato 0) |
| Metadados | 266 KB de cabeçalhos nos arquivos + índice de 55 KB | índice de 319 KB + 16 bytes por arquivo |
| Montar a partir das peças codificadas | 113 s | 181 s |
| Volta bit a bit dos 4 shards | 783 s | 838 s |
| Ler o modelo inteiro no estrato 2 (5,59 GB) | 376 s | 423 s |
| Baixar o modelo até o estrato `d` | 4 requisições por faixa de bytes, qualquer `d` | `d + 1` arquivos inteiros |
| Uma camada (a 16) nos estratos 0 / 2 / 7 | 2 / 4 / 9 requisições | 1 / 3 / 8 |
| Subir do estrato 0 para o 2 | 8 requisições (4 se o cliente guardou os cabeçalhos) | 2 arquivos inteiros |
| Um arquivo se lê sozinho | sim (abre no core e na página) | não: precisa do índice |

- **Tamanho e volta empatam**: o corpo é o mesmo byte a byte; a diferença de metadados é 0,003% do arquivo.
- **Tempo**: o D2 foi 7 a 12% mais lento para ler e voltar nesta rodada única, provavelmente por ler de 3 a 8 arquivos intercalados em vez de um em sequência. Não é diferença de formato, e uma rodada não separa isso de ruído de disco.
- **A diferença que importa é como se baixa.** No D2, escolher a precisão é escolher arquivos: `hf download --include "model-e0[0-2]-*"`, o cache do Hub, espelhos e torrents já fazem isso. No D1 é preciso um cliente que peça faixas de bytes de cada shard; o índice com os prefixos tira a ida extra para ler os cabeçalhos, mas não tira a necessidade do cliente. Rede de verdade não foi medida: as contagens acima saem das faixas de bytes que cada leitura precisa.

**Modelos maiores** (contagens do cabeçalho de cada shard, lido por requisição de faixa no Hub; tamanhos estimados com as taxas medidas no 8B, válidas só para BF16):

| Modelo | Shards | Tensores | Sem perdas | Arquivos D1 | Arquivos D2 (partes de 5 GB) | Tabela de streams |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B (medido) | 4 | 291 | 10,6 GB | 4 | 8 | 2.619 |
| Llama-3.1-70B | 30 | 723 | ≈ 93 GB | 30 | 21 | 6.507 |
| Qwen3-235B-A22B (MoE) | 118 | 36.945 | ≈ 311 GB | 118 | 64 | 332.505 |
| DeepSeek-V3 (MoE, FP8) | 163 | 91.991 | ≈ 574 GB (taxas do FP8 abaixo) | 163 | ≈ 119 | 1.378.559 |

- **Um `.strata` único** fica abaixo do teto de 500 GB por arquivo do Hugging Face até ≈ 378 bilhões de pesos BF16, e abaixo dos 200 GB recomendados até ≈ 150 bilhões. O Llama-3.1-405B sem perdas (≈ 536 GB) não cabe num arquivo. Mesmo onde cabe, um arquivo de 90 GB é pior de baixar e retomar do que vários de 5 GB, então dividir é o normal, não a exceção.
- **A tabela de streams é o problema dos MoE**, nos dois desenhos. Com os hashes da A.13, cada stream custa ≈ 150 bytes de JSON (78 medidos sem o `h`, mais 71 do hash). No DeepSeek-V3 seriam ≈ 205 MB, acima do teto de 100 MB que o leitor impõe ao cabeçalho (A.12). O D1 espalha isso em 163 cabeçalhos de ≈ 1,3 MB; o D2 junta tudo no índice. As duas correções estão logo abaixo.

**Escolha: D2 para distribuir, D1 como exportação.** Os dois têm o mesmo tamanho e a mesma volta, então o critério é o uso. O STRATA promete "a precisão é uma escolha de download", e o D2 entrega isso com arquivos comuns, sem cliente especial; subir de precisão é baixar mais arquivos sem tocar nos que já estão no disco. O D1 fica como formato de troca de um shard (abre sozinho no core, na página e no leitor em Rust). O custo do D2 é concentrar a confiança no índice: ele é o que precisa ser publicado e assinado (o SHA-256 do índice autentica todos os arquivos pelos `h` de cada stream), e nos MoE ele só cabe no teto do leitor com as duas correções abaixo.

**O que isto não prova.** Uma rodada só, num disco local; nenhum download real pelo Hub foi medido. Os tamanhos de 70B e 235B são extrapolação das taxas do 8B. O D2 não foi gravado em fluxo (o script monta os dois desenhos a partir das mesmas peças, para comparar os mesmos bytes), então o ganho de não precisar da segunda cópia é de desenho, não medido.

### 19.1 Tabela de streams nos MoE: tensores pequenos crus e tabela binária

Duas mudanças, implementadas juntas (spec A.5, A.12 e [B.9](02-especificacao-formato.md#b9-modelos-em-vários-arquivos)):

- **Tensores pequenos crus.** Um tensor de ponto flutuante com menos de 65.536 elementos vai inteiro num stream só, no estrato 0, com `n = null`. É uma opção do escritor (`cruAbaixo`), ligada no conversor de vários arquivos e desligada no core, para que os arquivos de um tensor só e os vetores públicos de antes não mudem. Os leitores JS e Rust aceitam o tensor cru; há um vetor válido e um hostil novos, e o fuzzing diferencial JS × Rust (4.000 casos) não teve divergência.
- **Tabela binária no D2.** A tabela de streams sai do índice JSON para o arquivo `model.strata.streams`, com registros de 68 bytes (36 sem hashes). O SHA-256 dela fica no índice, e o leitor valida cada registro.

**Tamanho da tabela**, contado nos cabeçalhos reais de cada shard (com hashes):

| Modelo | Streams antes | Com os pequenos crus | JSON antes | JSON com crus | Binário com crus |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-70B | 6.507 | 5.219 | 1,0 MB | 0,8 MB | 0,4 MB |
| Qwen3-235B-A22B | 332.505 | 329.489 | 49,5 MB | 49,1 MB | 22,4 MB |
| DeepSeek-V3 | 1.378.559 | 275.735 | **205 MB** | 41 MB | **18,7 MB** |

As duas mudanças atacam causas diferentes. No DeepSeek-V3 o volume vem das 45.867 escalas F32 de 25 streams cada, e os crus cortam 80% das linhas. No Qwen3-235B as linhas vêm de 36.945 tensores grandes de especialistas, os crus quase não mudam nada, e só o binário reduz (à metade). A lista de tensores do índice continua em JSON (nomes e formas): 19,4 MB no DeepSeek-V3 e 7,7 MB no Qwen3-235B, também abaixo do teto.

**De ponta a ponta num modelo FP8 real:** `Qwen/Qwen3-8B-FP8` (2 shards, 9,44 GB), o mesmo esquema do DeepSeek-V3 (pesos `F8_E4M3` com uma escala por bloco de 128 × 128), só que com as escalas em BF16. JSON: [`tests/resultados/shards__Qwen__Qwen3-8B-FP8.json`](../tests/resultados/shards__Qwen__Qwen3-8B-FP8.json).

| | |
|---|---|
| Streams | **4.851 → 1.675**: 397 escalas e normas pequenas viram 1 stream cada em vez de 9 |
| Tabela binária | 113.916 bytes (o JSON com hashes teria ≈ 250 KB; sem os crus, ≈ 720 KB) |
| Sem perdas | 7.447.913.581 bytes de corpo, **78,9%** do original |
| `F8_E4M3` (6,95 B de pesos) | **6,69 bits/peso** sem perdas: estrato 0 = 3,69, planos 1 a 3 ≈ 1,0 cada |
| `BF16` (embedding e `lm_head`, 1,24 B) | 10,55 bits/peso, como no Llama |
| Tensores crus | 1,46 MB originais, 0,90 MB gravados (o deflate continua valendo para eles) |
| Codificar | 1.517 s de CPU (2 shards em paralelo, ≈ 13 min); 5,4 GB de RAM no pico |
| Volta | os 2 shards com o SHA-256 do Hugging Face e o índice idêntico, nos dois desenhos, com hashes conferidos (D1 539 s, D2 545 s) |
| Ler até o estrato 3 (6,83 GB) | D1 454 s, D2 464 s: os pesos FP8 já saem exatos; o embedding e o `lm_head` com 3 planos |

- **Primeira medida de FP8 no projeto.** O FP8 já é um tipo de 8 bits, então sobra menos para comprimir: 83,6% do original nos pesos, contra 66% no BF16. O ganho vem quase todo do expoente de 4 bits, que carrega ≈ 2,7 bits de informação.
- **Num modelo misto, os estratos param de significar a mesma coisa para todos os tensores.** O estrato 3 já é o FP8 exato, mas ainda é o BF16 com 3 planos. Os arquivos `e04` a `e07` do D2 (155 MB cada) só têm o embedding e o `lm_head`.
- Com as taxas do Qwen3-8B-FP8, o DeepSeek-V3 sem perdas daria ≈ 574 GB (83% dos 688,6 GB), em ≈ 119 arquivos de até 5 GB no D2. É estimativa.

## 20. Nível 1 gravado como Q5_K, no llama.cpp sem mudança

Pergunta: o nível 1 do modo residual (seção 16), convertido para um GGUF `Q5_K` comum por [`strata_to_gguf.py`](../tests/ppl/strata_to_gguf.py), mantém a qualidade da emulação? ΔPPL numa T4 no wikitext-2 de teste inteiro, com os três GGUF desquantizados pelo mesmo caminho. Tabela em [`resultados/unsloth__Llama-3.2-1B_nivel1_q5k.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_nivel1_q5k.md).

| Configuração | download | RAM | ΔPPL |
|---|---:|---:|---:|
| Q4_K_M | 4,80 | 4,80 | +4,07% |
| Q5_K_M (feito à parte) | 5,66 | 5,66 | **+1,25%** |
| nível 1 emulado (seção 16) | 5,95 | 16 (emulação) | +1,17% |
| nível 1 gravado como Q5_K | 5,78 | 5,66 | +1,43% |

O download do nível 1 é o estrato 0 mais os bits do nível 1 lidos do arquivo residual (seção 15). A coluna "disco" da tabela do Colab mostra o tamanho do GGUF gerado, igual ao do Q5_K_M.

- **Convertido, o nível 1 fica um pouco atrás do Q5_K_M**: +1,43% contra +1,25%, com a mesma RAM e 0,13 bit a mais de download. Roda no llama.cpp sem mudança, mas não supera o Q5_K_M.
- **De onde vem a perda** (erro relativo dos pesos nas matrizes Q4_K das camadas 0, 5, 10 e 15): Q4_K 7,28%; nível 1 emulado 4,01%; mesma grade em Q5_K com o mínimo exato 4,03%; com o mínimo requantizado em 6 bits (o arquivo real) 4,23%; com `dmin` otimizado por busca 4,22%; Q5_K_M feito pelo llama-quantize 3,69%.
- **Os pesos saturados não explicam a vantagem** (ΔPPL numa T4, [`resultados/unsloth__Llama-3.2-1B_saturados.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_saturados.md)). Cortá-los na sub-célula da ponta, como o `Q5_K` faz, muda o nível 1 de +1,17% para +1,18% e o nível 2 de +0,308% para +0,313%. A hipótese anterior estava errada.
- **Sobra o mínimo requantizado** como causa: ele é a única diferença entre a emulação com saturados cortados (+1,18%, erro 4,03%) e o arquivo `Q5_K` (+1,43%, erro 4,23%). É inferência por eliminação, mas o resto do cálculo é o mesmo.
- A grade herdada do Q4_K erra mais nos pesos que a do Q5_K_M (4,03% contra 3,69%) e mesmo assim dá ΔPPL menor; erro médio dos pesos não ordena bem configurações tão próximas.
- **Bônus: os níveis ficam mais baratos.** Se os saturados não precisam vir exatos nos níveis, o arquivo adia o valor exato deles para o último nível ([`strata_residual_adiar.py`](../tests/ppl/strata_residual_adiar.py)). Nível 1: 5,95 → **5,84** bits/peso; nível 2: 6,95 → **6,91**; total sem perdas igual (11,24). Interpolando em log(ΔPPL), a qualidade do Q5_K_M sai com 5,79 bits (+0,13 sobre o Q5_K_M, antes +0,23) e a do Q6_K com 6,55 (−0,01: empate).
- **O que muda no plano.** O armazenamento continua igual: um arquivo de 11,24 bits/peso substitui BF16 + Q4 + Q5 + Q6. Gravar como Q5_K serve para rodar hoje, perdendo 0,18 ponto de ΔPPL para o Q5_K_M. O tipo próprio do kernel só precisa usar o mínimo do Q4_K sem requantizar (especificação B.8); não precisa de lista de saturados.

## 21. Modo residual gravado (`strata/0.3`)

Pergunta: as seções 15, 16 e 20 mediram o modo residual por estimativa de entropia, sem arquivo. Gravado de verdade (especificação [A.14](02-especificacao-formato.md#a14-modo-residual-strata03)), quanto ele custa, ele volta bit a bit para os dois originais e quanto demora para ler? Escrito e lido pelo leitor de referência em Rust (`strata-ref residual`, `leitor-rs/src/residual.rs`) a partir do `model.safetensors` do Hugging Face e do Q4_K_M do `llama-quantize` (llama.cpp 5aeac37, sem imatrix), com K = 3 níveis.

| Parte (bits por peso) | estrato 0 | níveis 1 a 3 | fecho | **sem perdas** | estimativa (seções 15 e 20) |
|---|---:|---:|---:|---:|---:|
| matrizes das camadas (973 M pesos) | 4,80 | 3 × 1,000 | 3,32 | **11,12** | 11,24 |
| matrizes Q4_K (830 M) | 4,50 | 3 × 1,000 | 3,61 | 11,11 | |
| matrizes Q6_K (405 M, embedding incluído) | 6,56 | 3 × 1,000 | 1,60 | 11,16 | |
| todas as matrizes (1.236 M) | 5,18 | 3 × 1,000 | 2,95 | 11,13 | |

- **O arquivo real é menor que a estimativa: 11,12 contra 11,24 bits/peso.** Os planos dos níveis vão crus, 1 bit exato por peso, e a estimativa cobrava cerca de 1,04 por nível. O fecho (o BF16 exato dentro da sub-célula do nível 3) custa 3,32, perto dos cerca de 3,30 estimados; 35.704 pesos (0,003%) vão por escape, crus.
- **Os níveis também ficam mais baratos.** Leituras parciais nas camadas: nível 1 = 5,80 bits/peso (estimado 5,84), nível 2 = 6,80 (6,91), nível 3 = 7,80. A regra de reconstrução é a da seção 20 (centro da sub-célula, saturados cortados na ponta), então os ΔPPL de lá devem valer para estes níveis: +1,18% no nível 1 e +0,31% no nível 2. Interpolando em log(ΔPPL), a qualidade do Q5_K_M sai com 5,75 bits (+0,09 sobre o Q5_K_M) e a do Q6_K com 6,46 (−0,10: abaixo do Q6_K). A interpolação é inferência: o ΔPPL não foi medido de novo sobre a saída do leitor, e rodar os níveis precisa dos tipos `Q4_K_R` e `Q6_K_R` do kernel (B.8; a seção 22 mostra que as matrizes Q6_K também precisam dos níveis).
- **Os dois originais voltam byte a byte.** A volta para `.safetensors` tem o SHA-256 do Hugging Face (68a2e4be…), e o estrato 0 remonta o GGUF Q4_K_M com o SHA-256 dele (002f464f…). Para o GGUF basta o prefixo de 801.942.795 bytes (o GGUF tem 807.690.144; as normas vão em BF16 e o arquivo as alarga para float32); com um byte a menos, o leitor recusa.
- **Tamanhos.** Arquivo de 1,72 GB, contra 1,64 GB do `.strata` 0.2 (+5%, o preço de começar num k-quant). Prefixos: estrato 0 802 MB, nível 1 956 MB, nível 2 1.111 MB, nível 3 1.265 MB.
- **Velocidade de leitura** (4 núcleos, neste contêiner, que é pessimista na primeira escrita em memória nova): decodificar e conferir tudo 25 s (o 0.2 leva 8,4 s); em fluxo 36 s; remontar o GGUF 8 s. O fecho decodifica cerca de 50 M pesos/s em 4 threads; cada peso faz uma busca binária, O(log c), sem laço que um arquivo hostil consiga alongar.
- **Blocos para o kernel.** `blocos_q4kr` e `blocos_q6kr` (Rust e Python) montam os tipos `Q4_K_R<k>` e `Q6_K_R<k>` (B.8) a partir do arquivo. Nas 112 matrizes do Llama-3.2-1B, com k = 1, 2 e 3, eles saem iguais bit a bit aos blocos que [`tests/kernel/test_q4kr.py`](../tests/kernel/test_q4kr.py) monta por conta própria a partir dos pesos BF16 (2,48 GB comparados, 0 bits diferentes), inclusive nos sub-blocos Q6_K de escala negativa.
- **Segurança.** As regras de A.12 e os hashes de A.13 valem inteiros. Os vetores públicos em [`tests/vetores/residual/`](../tests/vetores/residual/vetores.json) trazem um Llama sintético minúsculo, dois arquivos válidos e 15 hostis, um por regra de A.14; o leitor recusa todos, a leitura em fluxo dá o mesmo que a leitura normal em qualquer tamanho de pedaço, e em 2.000 mutações nenhum arquivo adulterado foi aceito com pesos errados.

## 22. Níveis só nas matrizes Q4_K (o que o tipo `Q4_K_R` cobre)

Pergunta: o Q4_K_M guarda 16 das 112 matrizes (15% dos pesos) em Q6_K, e o tipo `Q4_K_R<k>` do kernel (especificação B.8) só cobre as matrizes Q4_K. Se as Q6_K ficarem no estrato 0, a qualidade dos níveis se mantém? ΔPPL numa T4 no wikitext-2 de teste inteiro, saturados cortados em todas as linhas, com `strata_residual.py --so-q4k` ([notebook](../tests/ppl/strata_q4kr_colab.ipynb), tabela em [`resultados/unsloth__Llama-3.2-1B_so_q4k.md`](../tests/ppl/resultados/unsloth__Llama-3.2-1B_so_q4k.md)). Bits/peso nominais (planos crus de 1 bit), nas 112 matrizes.

| Configuração | bits/peso | ΔPPL |
|---|---:|---:|
| nível 1 só nas Q4_K (`Q4_K_R1`) | 5,656 | +1,32% |
| Q5_K_M | 5,656 | **+1,25%** |
| nível 1 em todas | 5,802 | +1,18% |
| nível 2 só nas Q4_K (`Q4_K_R2`) | 6,509 | +0,54% |
| Q6_K | 6,562 | **+0,49%** |
| nível 2 em todas | 6,802 | +0,31% |

- **No mesmo tamanho, o k-quant ganha por pouco.** O nível 1 só nas Q4_K tem exatamente os bytes do Q5_K_M e fica 0,07 ponto atrás. O nível 2 só nas Q4_K tem 0,05 bit a menos que o Q6_K e fica 0,05 ponto atrás.
- **As matrizes Q6_K importam.** Refiná-las custa 0,15 bit no nível 1 e tira 0,15 ponto de ΔPPL (de +1,32% para +1,18%); no nível 2, 0,29 bit e 0,22 ponto. Sem elas, os níveis perdem a vantagem que tinham na emulação.
- **Na mesma qualidade** (interpolando em log(ΔPPL) entre "só Q4_K" e "todas"): a do Q5_K_M sai com 5,72 bits (+0,07 sobre o Q5_K_M) e a do Q6_K com 6,56 (empate). É interpolação: o ponto intermediário seria refinar só parte das matrizes Q6_K, que não foi medido.
- **Consequência para o kernel:** precisa também de um tipo `Q6_K_R<k>` (bloco Q6_K + k planos), para que os níveis entreguem os +1,18% e +0,31% medidos. O Q6_K não tem mínimo, então o tipo é mais simples que o `Q4_K_R`. Para o arquivo 0.3, os níveis continuam valendo para todas as matrizes (`niveis` do arquivo inteiro), como já está.
- Um modelo só (1B, sem imatrix): diferenças de 0,05 a 0,07 ponto são reais nesse texto, mas pequenas; a conferência num modelo maior continua pendente.

## 23. STRATA dentro do llama.cpp

Pergunta: com os tipos dos blocos montados dentro do ggml (patch em [`kernel/ggml/`](../kernel/ggml/), [kernel/README.md](../kernel/README.md#dentro-do-llamacpp)), os níveis 1 e 2 rodam no `llama-perplexity` e no `llama-bench` com a qualidade da emulação e na velocidade dos k-quants? Llama-3.2-1B, GGUF dos níveis feito por [`strata_gguf_niveis.py`](../tests/kernel/strata_gguf_niveis.py) a partir do Q4_K_M e do BF16, k-quants do `llama-quantize` sem imatrix. Colab Pro, CPU AMD EPYC 9B14 (22 núcleos físicos, AVX-512), [notebook](../tests/kernel/strata_llamacpp_colab.ipynb), tabela completa em [`resultados/unsloth__Llama-3.2-1B_llamacpp_epyc.md`](../tests/kernel/resultados/unsloth__Llama-3.2-1B_llamacpp_epyc.md). Perplexidade no texto de teste inteiro do wikitext-2 na cópia não-raw do PyTorch (o download do raw falhou nessa rodada; vale para comparar os modelos entre si), KL contra o BF16 nos 100 primeiros trechos, tok/s = mediana de 3 rodadas alternadas.

| modelo | GGUF (GiB) | ΔPPL | KL (×1000) | mesmo 1º token | geração, 1 thread | geração, 22 threads | prompt 512, 22 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 0,752 | +3,08% | 40,7 | 89,1% | 27,7 | 117,1 | 1.245 |
| Q5_K_M | 0,849 | +1,35% | **13,8** | 93,7% | 19,0 | 102,8 | 1.696 |
| STRATA nível 1 | 0,882 | **+0,67%** | 14,1 | 93,4% | 19,1 | 99,8 | 513 |
| Q6_K | 0,952 | +0,51% | 7,0 | 95,2% | 24,7 | 94,1 | 1.526 |
| STRATA nível 2 | 0,979 | **+0,32%** | **5,1** | 95,9% | 18,7 | 91,2 | 505 |

- **A qualidade da emulação se confirma dentro do llama.cpp.** O nível 1 empata com o Q5_K_M: a KL fica 2% acima (dentro de uns 2 erros-padrão) e a perplexidade fica melhor. O nível 2 passa o Q6_K: KL 27% menor, 96% de primeiro token igual contra 95%. A KL é a medida a olhar aqui; a perplexidade num texto deste tamanho e com estas diferenças pequenas é menos confiável (o ΔPPL do nível 1 sai bem melhor que o da emulação, +0,67% contra +1,18%, e a KL diz que ele é só igual ao Q5_K_M).
- **Na RAM, o STRATA gasta um pouco mais:** +3,9% de GGUF no nível 1 e +2,9% no nível 2. É o preço de começar no Q4_K_M (o mínimo não requantizado e o Q6_K_P1 com 1 bit a mais nas matrizes Q6_K). A vantagem continua sendo o armazenamento: um arquivo de 11,1 bits/peso no lugar de BF16 + Q4 + Q5 + Q6 (seção 21).
- **Geração de texto empata:** 0,97 a 1,00 do k-quant de mesma qualidade com 1 e com 22 threads, a não ser o nível 2 numa thread só (0,76 do Q6_K; ali a conta pesa mais que a memória, como no microbenchmark do kernel).
- **A leitura do prompt ficou em 0,30 a 0,33 dos k-quants.** Causa: com várias linhas de entrada de uma vez, o llama.cpp usa uma "linha de montagem" própria para os k-quants (o caminho *tiled* do ggml-cpu, que multiplica blocos de 16 × 16 de uma vez), e os tipos do STRATA caíam no produto escalar linha a linha.
- **Correção (depois desta rodada):** os quatro tipos passam pelo caminho tiled. Cada bloco montado vira os códigos sem sinal que o tile já sabe ler: no Q4_K_P, código 2u + 1 com deslocamento 2^k e o mínimo do Q4_K; no Q6_K_P, código w + 128 com deslocamento 128 e o meio-passo no lugar do mínimo. Conferido contra o produto escalar com pesos reais e ativações reais do modelo, nas versões AVX-512 VNNI e AVX2: diferença máxima de 4e-7 da saída. No modelo inteiro a KL contra o BF16 é a mesma com e sem o caminho novo (nível 1: 14,7 ± 0,5 contra 14,5 ± 0,5; nível 2: 5,37 ± 0,17 contra 5,31 ± 0,15, 8 trechos no container). A perplexidade de 8 trechos muda até 0,4% entre os dois: a diferença mínima de arredondamento muda a quantização das ativações para 8 bits na camada seguinte, e esse ruído não tem direção. No container, a leitura do prompt dos níveis ficou 2,5 a 4,5 vezes mais rápida com o caminho novo; a comparação com os k-quants no container varia demais entre rodadas e fica para o Colab.
- **Com o caminho novo, a leitura do prompt empata** (segunda rodada no Colab, CPU Intel Xeon de 6 núcleos físicos com AVX-512, só velocidade, [`resultados/unsloth__Llama-3.2-1B_llamacpp_xeon_tiled.md`](../tests/kernel/resultados/unsloth__Llama-3.2-1B_llamacpp_xeon_tiled.md)). Relação com o k-quant de mesma qualidade, mediana de 3 rodadas alternadas (cada rodada variou menos de 1%):

  | | geração, 1 thread | geração, 6 threads | geração, 12 threads | prompt 512, 6 threads |
  |---|---:|---:|---:|---:|
  | nível 1 / Q5_K_M | 0,97 | 0,98 | 0,98 | **0,94** (325 contra 346 tok/s) |
  | nível 2 / Q6_K | 0,86 | 0,88 | 0,90 | **1,01** (324 contra 321) |

  O nível 1 fica a 2–6% do Q5_K_M em tudo. O que sobra é a geração do nível 2, 10–14% mais lenta que o Q6_K nesta CPU (no EPYC da primeira rodada eram 3% com 22 threads): na RAM o bloco Q4_K_P2 tem as escalas de 6 bits e o mínimo do Q4_K para desembrulhar, e o Q6_K_P ocupa 8,56 bits/peso contra 6,56 do Q6_K nas matrizes Q6_K. As duas rodadas foram em máquinas diferentes; só as razões dentro de cada rodada se comparam.
- Um modelo só (1B). Falta repetir tudo num 8B.

## 24. Llama-3.1-8B na GPU, com e sem imatrix

Pergunta: o resultado do 1B (nível 1 ≈ Q5_K_M, nível 2 melhor que o Q6_K) se repete num modelo maior, e contra os k-quants feitos com imatrix, que é como quase todo GGUF publicado é feito? E os tipos rodam na GPU? Llama-3.1-8B, llama.cpp com o patch e os kernels CUDA de [`kernel/cuda/`](../kernel/cuda/) ([kernel/README.md](../kernel/README.md#na-gpu-cuda)), Colab Pro com NVIDIA A100-SXM4-80GB, [notebook](../tests/kernel/strata_8b_gpu_colab.ipynb), tabela completa em [`resultados/unsloth__Meta-Llama-3.1-8B_llamacpp_a100.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_llamacpp_a100.md). imatrix de 125 trechos de 512 tokens do wikitext-2 raw de *treino*; perplexidade no texto de teste inteiro, KL contra o BF16 nos 100 primeiros trechos; tok/s = mediana de 3 rodadas alternadas, todas as camadas na GPU. O teste CUDA × CPU (`strata_cuda_check`) e a conferência do caminho de geração no modelo inteiro passaram antes das medidas.

| modelo | GGUF (GiB) | ΔPPL | KL (×1000) | KL (×1000), com imatrix | geração (tok/s) | prompt 512 (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 4,583 | +2,77% | 29,47 | 26,26 | 153,9 | 4.714 |
| Q5_K_M | 5,339 | +1,00% | 10,18 | **8,60** | 144,6 | 4.570 |
| STRATA nível 1 | 5,512 | +0,86% | **9,98** | 10,78 | 144,0 | 6.808 |
| Q6_K | 6,143 | +0,34% | 4,28 | **3,47** | 117,4 | 4.255 |
| STRATA nível 2 | 6,208 | +0,23% | **3,56** | 3,66 | 141,0 | 6.792 |

(o STRATA "com imatrix" é o de sempre feito sobre o Q4_K_M com imatrix; tok/s só dos modelos sem imatrix, que têm o mesmo tamanho)

- **Sem imatrix, o 1B se repete:** nível 1 empata com o Q5_K_M (KL 2% menor, +3,2% de GGUF), nível 2 fica 17% mais perto do original que o Q6_K com +1,1% de GGUF.
- **Com imatrix, o STRATA perde nos dois níveis.** A imatrix melhora os k-quants em 16–19% de KL e piora o STRATA feito sobre o Q4_K_M com imatrix (nível 1 +8%, nível 2 +3%). Melhor contra melhor: nível 1 sem imatrix 9,98 contra Q5_K_M com imatrix 8,60 (+16%); nível 2 sem imatrix 3,56 contra Q6_K com imatrix 3,47 (+3%). A vantagem de qualidade do modo residual valia só contra k-quants sem imatrix.
- **Por quê (hipótese, confirmada no 1B na seção 25):** a imatrix escolhe as escalas de cada sub-bloco para que os pesos importantes caiam perto das 16 marcas do Q4. Os níveis dividem cada intervalo em 2 ou 4 de forma fixa, e esse alinhamento não vale mais na grade fina.
- **Correção (seção 28):** na GPU, os tipos P medidos aqui não arredondam as ativações na leitura em lote e os k-quants sim. Os "ganhos" sem imatrix (9,98 contra 10,18; 3,56 contra 4,28) e as diferenças da linha com imatrix vêm em parte disso.
- **Na GPU, a geração empata ou ganha:** nível 1 a 1,00 do Q5_K_M, nível 2 a 1,20 do Q6_K (o Q6_K do llama.cpp desembrulha 6 bits espalhados em dois campos; o Q6_K_P lê int8 direto).
- **A leitura do prompt saiu 1,5–1,6 vez mais rápida que nos k-quants, mas isso não deve ser mérito do formato:** os tipos do STRATA não têm o caminho MMQ e caem na desquantização + cuBLAS em FP16, que na A100 foi mais rápida que o MMQ dos k-quants. Para separar formato de caminho, falta medir os k-quants com `GGML_CUDA_FORCE_CUBLAS`.

## 25. Base fina: ajustar a grade no nível de cima e tirar o Q4 por corte

Pergunta: o STRATA perde para os k-quants com imatrix (seção 24) porque os níveis subdividem um Q4_K que a imatrix ajustou para 16 marcas. E se a imatrix ajustar a grade fina e o Q4 sair dela? [`strata_base_fina.c`](../tests/kernel/strata_base_fina.c) / [`.py`](../tests/kernel/strata_base_fina.py): cada sub-bloco das matrizes Q4_K é quantizado direto na grade de 4 + T bits (mesma busca do `quantize_row_q4_K_impl` do llama.cpp, com a imatrix, trocando 15 por 2^(4+T) − 1), as escalas e os mínimos de 6 bits saem como no Q4_K, e q4 = u >> T, planos = os T bits baixos de u. O arquivo continua o mesmo: o nível 0 é um Q4_K comum e os níveis usam os tipos de sempre. Opcionalmente a escolha da escala soma o erro dos níveis 0 … T com pesos (`--pesos`). As matrizes Q6_K ficam como antes. Conferência: com T = 0 o programa dá o Q4_K_M com imatrix do `llama-quantize` byte a byte.

Llama-3.2-1B no container, KL contra o BF16 em 40 trechos do wikitext-2 raw, imatrix de 125 trechos do texto de treino; tabela completa em [`resultados/unsloth__Llama-3.2-1B_base_fina_container.md`](../tests/kernel/resultados/unsloth__Llama-3.2-1B_base_fina_container.md). KL ×1000:

| nível | k-quant com imatrix | STRATA sobre o Q4_K_M com imatrix | base fina T=2, pesos 1, 2, 4 | base fina T=1 | base fina T=2, só o nível 2 |
|---|---:|---:|---:|---:|---:|
| 0 (Q4_K_M, 0,752 GiB) | **40,9** | 40,9 | 55,4 | 54,2 | 61,7 |
| 1 (Q5_K_M 0,849 / STRATA 0,882 GiB) | 13,9 | 17,7 | 14,3 | **12,5** | 15,6 |
| 2 (Q6_K 0,952 / STRATA 0,979 GiB) | 7,3 | 6,4 | 5,4 | | **5,3** |

- **A hipótese se confirma:** ajustar na grade fina recupera o que a imatrix dá. O nível 1 vai de +28% contra o Q5_K_M com imatrix para +3% (T=2, pesos 1, 2, 4) ou −10% (T=1); o nível 2 vai de −12% para −26% contra o Q6_K com imatrix.
- **O preço é o nível 0:** +32% a +51% de KL contra o Q4_K_M com imatrix, pior até que o Q4_K_M sem imatrix (47,7). Pesar o erro do nível 0 na escolha da escala quase não ajuda (61,7 → 55,4): o custo é da estrutura, não da busca. As 16 marcas do Q4 são os centros de grupos de 4 marcas finas, então ficam 1,5 passo fino para dentro da faixa em cada ponta, e os pesos das pontas são cortados mais cedo.
- **Onde fica o nível 0:** ainda bem mais perto do Q4_K_M que do Q3_K_M com imatrix (130,1 com 0,643 GiB). Interpolando o log da KL entre os dois, 55 equivale a um k-quant de ~4,6 bits/peso no lugar de 4,8 (estimativa, não medida).
- **Não dá para um arquivo aninhado ser ótimo em todos os níveis ao mesmo tempo.** A base define qual nível é "de primeira": o Q4 (como até agora, e aí os níveis 1 e 2 perdem para os k-quants com imatrix) ou os níveis de cima (base fina, e aí o Q4 fica ~0,2 bit pior). Só o 1B foi medido; o [notebook](../tests/kernel/strata_8b_fina_colab.ipynb) repete no Llama-3.1-8B na GPU.

**No Llama-3.1-8B a base fina se confirma** (Colab A100, caminho real dos tipos P no CUDA, wikitext-2 raw de teste inteiro para a PPL, KL contra o BF16 em 100 trechos, [`resultados/unsloth__Meta-Llama-3.1-8B_base_fina_a100.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_base_fina_a100.md)). É a primeira medida no 8B em que os níveis 1 e 2 do STRATA não perdem para os k-quants com imatrix:

| nível | modelo | GGUF (GiB) | ΔPPL | KL ×1000 | contra o k-quant |
|---|---|---:|---:|---:|---:|
| 0 | Q4_K_M com imatrix | 4,583 | +2,57% | **26,26** | |
| 0 | base fina T=2, pesos 1, 2, 4 | 4,583 | +3,24% | 34,24 | +30% |
| 0 | base fina T=1 | 4,583 | +3,43% | 33,11 | +26% |
| 1 | Q5_K_M com imatrix | 5,339 | +0,81% | 8,60 | |
| 1 | base fina T=2, pesos 1, 2, 4 | 5,512 | +0,80% | 8,69 | +1% |
| 1 | base fina T=1 | 5,512 | +0,78% | **7,59** | −12% |
| 2 | Q6_K com imatrix | 6,143 | +0,31% | 3,47 | |
| 2 | base fina T=2, pesos 1, 2, 4 | 6,208 | +0,26% | **3,16** | −9% |

- **Correção (seção 28):** esta comparação usa os k-quants do caminho real, que arredondam as ativações; os tipos P na GPU não. No mesmo caminho (k-quants em F16: Q5_K_M 7,55, Q6_K 3,31), o nível 1 com T=1 empata com o Q5_K_M ocupando 3,2% a mais, e o nível 2 com T=2 empata com o Q6_K. A frase "primeira medida em que não perdem" e as razões −12% / −9% valem só para o caminho real.
- **No mesmo tamanho** (interpolando o log da KL entre Q5_K_M e Q6_K com imatrix; estimativa, não medida): um k-quant de 5,512 GiB teria KL ~7,07, então o nível 1 com T=1 fica ~7% atrás; um de 6,208 GiB teria ~3,23 (extrapolado), e o nível 2 com T=2 fica ~2% à frente, ou seja, empate. Antes da base fina, a diferença no mesmo tamanho era de ~25% ou mais no nível 1.
- **O empacotamento do Q6_K_P1 vira a diferença no nível 1.** Pelos tamanhos dos arquivos, o 8B tem ~1,0 bilhão de pesos em matrizes Q6_K e ~6,0 bilhões em Q4_K. Guardar o Q6_K_P1 empacotado (7,56 em vez de 8,56 bits/peso) tira 0,117 GiB: o nível 1 iria para 5,395 GiB (+1,0% contra o Q5_K_M), onde o k-quant interpolado teria ~8,07, e o T=1 (7,59) ficaria ~6% à frente. O nível 2 não muda (o Q6_K_P2 já usa os 8 bits).
- **O preço do nível 0 é o mesmo do 1B:** +26 a +30% de KL. Pela curva entre Q3_K_M e Q4_K_M com imatrix, equivale a um k-quant ~0,18 GiB menor (~0,22 bit/peso nas matrizes das camadas). O nível 0 continua um Q4_K comum, que roda no llama.cpp sem patch, mas deixa de ser igual byte a byte ao Q4_K_M publicado nos hubs.
- **Nenhuma das duas versões é boa nos três níveis:** T=1 ganha no nível 1 e não tem nível 2 medido; T=2 com pesos 1, 2, 4 ganha no nível 2 e só empata no nível 1. Um arquivo tem uma base só.
- **Versão única, triagem no 1B** (container, 40 trechos, KL ×1000; `strata_base_fina.py --K` grava níveis acima de T dividindo ao meio a célula da grade fina). T=1 com nível 2: 54,2 / 12,45 / 6,11; T=2 com pesos 1, 4, 2 (nível 1 pesando mais): 55,7 / 14,4 / 5,6, sem ganho no nível 1 sobre os pesos 1, 2, 4 (14,3). O nível 1 da T=2 fica preso à grade de 6 bits, não à busca. A candidata é a T=1 com nível 2 (no 1B, −10% contra o Q5_K_M e −16% contra o Q6_K com imatrix). A razão do nível 1 passou do 1B ao 8B quase igual (0,90 → 0,88), a do nível 2 não (0,74 → 0,91); se o nível 2 da T=1 piorar na mesma proporção, ele empata com o Q6_K no 8B (estimativa). O [notebook](../tests/kernel/strata_8b_fina_t1_colab.ipynb) mede isso no 8B, com o Q8_0 na tabela para ter a curva dos k-quants acima do Q6_K.

## 26. Níveis com escala própria: o Q4 fica intacto

A base fina (seção 25) melhora os níveis 1 e 2 estragando o nível 0, e o nível 0 é o que roda em hardware com pouca memória. Alternativa: o nível 0 continua sendo o Q4_K_M com imatrix, e cada nível guarda, além do bit por peso, o tamanho do passo. Hoje o nível k anda ±passo/2^(k+1) a partir do nível anterior (o centro da metade da célula); aqui anda ±m·passo/2^(k+4), com m de 0 a 15 escolhido com a imatrix (m = 8 é o de hoje). [`strata_escala_nivel.py`](../tests/kernel/strata_escala_nivel.py) monta os pesos e grava as matrizes em F16, para medir a qualidade sem kernel novo; com m = 8 os pesos saem iguais bit a bit aos dos tipos P. Em F16 o llama.cpp não arredonda as ativações para 8 bits, e a KL de todos cai uns 2 (×1000); por isso os k-quants também foram medidos em F16. Llama-3.2-1B, 40 trechos, tabela em [`resultados/unsloth__Llama-3.2-1B_escala_nivel_container.md`](../tests/kernel/resultados/unsloth__Llama-3.2-1B_escala_nivel_container.md). KL ×1000:

| | bits extras por nível | nível 1 | nível 2 |
|---|---:|---:|---:|
| Q5_K_M / Q6_K com imatrix | | 11,79 | 5,37 |
| STRATA de hoje (m = 8) | 0 | 15,87 | 4,43 |
| m = 7 para todos | 0 | 13,94 | 4,20 |
| m por bloco de 256 | 0,016 | 12,41 | 4,20 |
| m por sub-bloco | 0,143 | **11,68** | **4,05** |

- **O passo fixo de hoje é grande demais.** O m escolhido fica quase sempre entre 5 e 8, com pico em 6 e 7. Com a imatrix, os pesos importantes caem perto das marcas do Q4, então o erro que sobra é menor que meia célula. Só trocar a constante para m = 7 (sem nenhum bit a mais) leva o nível 1 de +35% para +18% contra o Q5_K_M com imatrix e o nível 2 de −18% para −22% contra o Q6_K.
- **Uma escala por bloco de 256 pesos (0,016 bit/peso por nível) pega quase todo o ganho:** nível 1 a +5% do Q5_K_M com imatrix, nível 2 22% melhor que o Q6_K com imatrix, e o nível 0 continua o Q4_K_M com imatrix, igual byte a byte.
- **Contra a base fina (seção 25):** nos níveis 1 e 2 as duas ficam parecidas (projetando a soma de ~2 das ativações, bloco ≈ 14,3 e 6,2 contra 14,3 e 5,4 da base fina; estimativa), mas o nível 0 aqui não perde nada (40,9 contra 55,4).
- **No tamanho:** o nível 1 ocupa 3,9% a mais de RAM que o Q5_K_M, quase tudo por causa do Q6_K_P1 montado em int8 (8,56 bits/peso nas matrizes Q6_K, contra 7,56 se ficasse empacotado). Nessa diferença de tamanho, um k-quant ganharia uns 20% de KL (interpolando entre Q5_K_M e Q6_K), então o nível 1 ainda fica atrás no mesmo tamanho; o nível 2 (+2,9% de RAM, −22% de KL) fica um pouco à frente.
- **No Llama-3.1-8B o ganho encolhe e o STRATA continua atrás** ([notebook](../tests/kernel/strata_8b_escala_colab.ipynb), Colab L4, mesmo caminho F16, KL contra o BF16 em 100 trechos, [`resultados/unsloth__Meta-Llama-3.1-8B_escala_nivel_l4.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_escala_nivel_l4.md)). KL ×1000:

  | | nível 1 | contra Q5_K_M | nível 2 | contra Q6_K |
  |---|---:|---:|---:|---:|
  | k-quant com imatrix | 7,48 | | 3,25 | |
  | STRATA de hoje (m = 8) | 10,72 | +43% | 3,65 | +12% |
  | m por bloco de 256 | 8,81 | +18% | 3,48 | +7% |

  A escala por bloco ajuda nos dois modelos (nível 1 −18% no 8B, −22% no 1B), mas no 8B não chega aos k-quants com imatrix em nenhum nível, e ainda ocupa 3–4% a mais de RAM. É a segunda vez que o 1B promete mais do que o 8B entrega (a primeira foi a imatrix, seção 24): no 1B o Q6_K com imatrix é relativamente pior, e o nível 2 do STRATA parecia ganhar dele.
- **Leitura:** com imatrix, todo nível do STRATA sobre o Q4_K_M custa mais bits que o k-quant de mesma qualidade (no 8B, o nível 1 com escala está a uns 18% de KL e 3,2% de RAM do Q5_K_M). O preço é de estar aninhado: cada nível só pode refinar a grade do anterior. O que o formato oferece em troca continua sendo o armazenamento: um arquivo de ~11 bits/peso no lugar de BF16 + Q4 + Q5 + Q6 (seção 21). A base fina medida depois no 8B (seção 25) muda essa leitura para os níveis 1 e 2, ao custo do nível 0.

## 27. Três vertentes para os níveis: triagem no 1B e notebook do 8B

Plano e pesquisa em [`07-quantizacao-aninhada.md`](07-quantizacao-aninhada.md). As três vertentes, conferidas no Llama-3.2-1B (container, KL ×1000 contra o BF16 em 40 trechos, imatrix de 125 trechos):

- **1. Escala conjunta** (`strata_base_fina.py --conjunta A`): além da busca na grade fina, faz a busca de 4 bits do Q4_K_M e testa 11 pontos na reta entre as duas (escala e mínimo), ficando com o de menor erro somado dos níveis 0, 1 e 2. O peso do nível 1 é A vezes o do nível 0, e o do nível 2 é A/2, já descontado o passo menor de cada nível. O nível 0 continua um Q4_K comum e os níveis usam os tipos P de hoje.
- **2. Base fina T=1 com nível 2** (`--T 1 --K 2`, seção 25).
- **3. Sub-células não uniformes** (`strata_escala_nivel.py --modo desloc`): o nível 0 é o Q4_K_M com imatrix, intacto. No nível k, os dois filhos de cada código do nível k − 1 ganham deslocamentos próprios (0 a 31, em unidades de passo/2^(k+4)), escolhidos com 3 rodadas de Lloyd ponderado pela imatrix, por linha ou por tensor. Só emulação em F16, por enquanto.

| | caminho | nível 0 | nível 1 | nível 2 |
|---|---|---:|---:|---:|
| k-quants com imatrix (Q4_K_M / Q5_K_M / Q6_K) | real | 40,9 | 13,9 | 7,3 |
| STRATA sobre o Q4_K_M com imatrix (seção 25) | real | 40,9 | 17,7 | 6,4 |
| 1. escala conjunta, A = 0 | real | 40,3 | 17,5 | 6,3 |
| 1. escala conjunta, A = 0,25 | real | 42,6 | 15,7 | 5,8 |
| 1. escala conjunta, A = 1 | real | 47,3 | 14,2 | 5,6 |
| base fina T=2, pesos 1, 2, 4 (seção 25) | real | 55,4 | 14,3 | 5,4 |
| 2. base fina T=1 com nível 2 | real | 54,2 | **12,5** | 6,1 |
| k-quants com imatrix em F16 (Q5_K_M / Q6_K) | F16 | | 11,8 | 5,4 |
| STRATA de hoje (passo fixo) | F16 | | 15,9 | 4,4 |
| 3. sub-células por tensor (sem custo) | F16 | 40,9 | 13,1 | 4,1 |
| 3. sub-células por linha (+0,07 e +0,22 bit/peso acumulados) | F16 | 40,9 | 11,7 | 3,5 |
| 1 + 3: base da escala conjunta A = 1, sub-células por linha | F16 | 47,3 | **10,2** | **3,1** |

- **A escala conjunta domina a base fina T=2:** com A = 1, o nível 0 perde 16% em vez de 35% e os níveis 1 e 2 ficam iguais (14,2 contra 14,3; 5,6 contra 5,4). A = 0 é o Q4_K_M com uma busca um pouco melhor (40,3 contra 40,9).
- **As sub-células não uniformes mexem nos níveis de cima sem tocar no nível 0:** por linha, o nível 1 empata com o Q5_K_M e o nível 2 fica 34% melhor que o Q6_K (F16), pagando 0,07 e 0,22 bit/peso de tabelas no 1B. No 8B as linhas são 2 a 7 vezes mais longas, e as tabelas pesam proporcionalmente menos.
- **A combinação 1 + 3 é a melhor nos níveis de cima** (−13% e −42% contra Q5_K_M e Q6_K em F16), ao custo de 16% no nível 0.
- **Dois erros encontrados na escala conjunta, corrigidos antes destas medidas:**
  - a busca podia escolher um mínimo de Q4_K negativo, que não pode ser gravado; agora cada candidato é avaliado com o mínimo já preso em 0;
  - a busca de 4 bits zerava, por um instante, os pesos dos níveis guardados num estado global e compartilhado entre threads. O estado agora é por thread. O modo sem `--conjunta` nunca muda esse estado, então as medidas anteriores da base fina valem.
- **Ressalva de sempre:** o 1B já prometeu mais do que o 8B entregou duas vezes (seções 24 e 26). O [notebook](../tests/kernel/strata_8b_vertentes_colab.ipynb) mede as quatro opções no Llama-3.1-8B (A100), com os k-quants Q3_K_M a Q8_0 em cada caminho, velocidade na GPU e a régua de `07-quantizacao-aninhada.md` calculada no fim.

## 28. Três vertentes no Llama-3.1-8B: nenhuma ganha no mesmo tamanho

Colab A100-80GB, [notebook](../tests/kernel/strata_8b_vertentes_colab.ipynb), KL contra o BF16 em 100 trechos do wikitext-2 raw de teste, imatrix de 125 trechos do texto de treino, tabela completa em [`resultados/unsloth__Meta-Llama-3.1-8B_vertentes_a100.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_vertentes_a100.md). KL ×1000 (erro padrão ~2%):

| | nível 0 | nível 1 | nível 2 |
|---|---:|---:|---:|
| k-quants com imatrix, caminho real (Q4_K_M / Q5_K_M / Q6_K; Q3_K_M 77,94, Q8_0 0,79) | 26,26 | 8,60 | 3,47 |
| k-quants com imatrix em F16 (Q8_0 0,57) | 25,32 | 7,55 | 3,31 |
| STRATA de hoje (passo fixo) | 26,26 | 10,78 | 3,66 |
| 1. escala conjunta, A = 0,5 | 28,31 | 8,87 | 3,24 |
| 1. escala conjunta, A = 1 | 29,95 | 9,09 | 3,29 |
| 1. escala conjunta, A = 2 | 31,25 | 8,30 | 3,28 |
| 2. base fina T=1 com nível 2 | 33,11 | **7,59** | 3,57 |
| 3. sub-células por tensor (sem custo) | 26,26 | 9,03 | 3,35 |
| 3. sub-células por linha (+0,037 e +0,074 bit/peso) | 26,26 | 8,49 | 3,14 |
| 1 + 3: base A = 1, sub-células por linha | 29,95 | 8,12 | **3,05** |

**Os tipos P na GPU não arredondam as ativações na medida da KL.** O `llama-perplexity` lê lotes de 512 tokens; nesse tamanho os k-quants usam o MMQ, que arredonda as ativações para 8 bits, e os tipos P caem na desquantização para FP16 + cuBLAS, que não arredonda. A prova está na tabela: o STRATA de hoje emulado em F16 dá 10,78 / 3,66, exatamente o que os tipos P deram no caminho real (seção 24). Comparar os níveis 1 e 2 com os k-quants do caminho real dá ao STRATA uma vantagem que é do caminho, não do formato: os k-quants em F16 ficam 4–12% melhores. Na geração (lotes de até 8) e na CPU, os dois lados arredondam. Por isso os níveis 1 e 2 abaixo são comparados com a curva dos k-quants em F16, e o nível 0 (Q4_K comum dos dois lados) com a curva do caminho real.

No mesmo tamanho de RAM, em bits por peso a mais que o k-quant com imatrix de mesma KL (interpolando o log da KL; o nível 1 já com o Q6_K_P1 empacotado, 0,117 GiB a menos; as tabelas da vertente 3 somadas):

| | nível 0 | nível 1 | nível 2 | soma |
|---|---:|---:|---:|---:|
| STRATA de hoje | 0 | +0,34 | +0,20 | +0,54 |
| 1. escala conjunta, A = 0,5 | +0,07 | +0,19 | **+0,06** | 0,32 |
| 1. escala conjunta, A = 2 | +0,17 | +0,14 | +0,07 | 0,38 |
| 2. base fina T=1 com nível 2 | +0,22 | **+0,07** | +0,17 | 0,46 |
| 3. sub-células por tensor | **0** | +0,21 | +0,10 | **0,31** |
| 3. sub-células por linha | **0** | +0,20 | +0,13 | 0,33 |
| 1 + 3 | +0,13 | +0,16 | +0,09 | 0,38 |

- **Nenhuma vertente empata com os k-quants com imatrix no mesmo tamanho, em nenhum nível.** Todas melhoram o STRATA de hoje: a soma cai de 0,54 para 0,31–0,46 bit/peso.
- **Cada vertente ganha num nível e paga em outro:** a base fina T=1 tem o melhor nível 1 (7,59, quase o Q5_K_M em F16, +0,07 bit) e paga 0,22 bit no nível 0 e 0,17 no nível 2; a escala conjunta A = 0,5 tem o melhor nível 2 (+0,06) e paga no nível 1.
- **A vertente 3 por tensor tem a menor soma e não mexe no nível 0** (igual byte a byte ao Q4_K_M com imatrix do hub): +0,21 bit no nível 1 e +0,10 no nível 2, sem tabela que pese. Por linha não compensa: o ganho de KL no nível 1 é comido pelas tabelas, e o nível 2 fica pior.
- **A combinação 1 + 3 não soma os ganhos do 1B:** no 1B ela era a melhor nos níveis 1 e 2 (seção 27); no 8B fica no meio.
- **Velocidade na GPU:** geração do nível 1 a 1,00 do Q5_K_M, nível 2 a 1,20 do Q6_K (passam no mínimo de 0,95). O prompt 1,46–1,59× mais rápido é o cuBLAS contra o MMQ, não o formato.
- **O 1B prometeu mais pela quarta vez.** Ali a medida foi na CPU, onde os dois lados arredondam as ativações, então a diferença é do modelo.
- **Leitura:** o preço de estar aninhado numa grade uniforme ficou em ~0,1–0,2 bit/peso por nível, como nos trabalhos da [pesquisa](07-quantizacao-aninhada.md) (AnyBCQ, MatGPTQ). O que o formato oferece em troca é o armazenamento: um arquivo de ~11 bits/peso no lugar de BF16 + Q4 + Q5 + Q6 (seção 21), com o estrato 0 igual ao Q4_K_M do hub.

## 29. Sub-células por tensor dentro do llama.cpp (tipos D), 1B na CPU

A vertente 3 por tensor virou os tipos `STRATA_Q4_K_D1/D2` e `STRATA_Q6_K_D1/D2` (ids 47–50, especificação B.8, [`kernel/README.md`](../kernel/README.md)): os blocos dos tipos P mais uma tabela de 16 bytes por tensor no GGUF. Junto, o Q6_K_P1 passou a usar o próprio bloco Q6_K_R1 na RAM (242 bytes em vez de 274): o nível 1 cai de 5,95 para 5,80 bits/peso nas matrizes das camadas do Llama-3.2-1B, sem mudar nenhum valor.

**Qualidade** (Llama-3.2-1B, CPU, caminho real dos dois lados, 40 trechos do wikitext-2, KL ×1000 contra o BF16, imatrix nos k-quants):

| | nível 1 | nível 2 |
|---|---:|---:|
| k-quant com imatrix (Q5_K_M 5,66 b / Q6_K 6,56 b) | 13,87 | 7,28 |
| tipos P (5,80 / 6,80 b) | 17,75 | 6,40 |
| tipos D, sub-células por tensor (5,80 / 6,80 b) | 15,37 | 5,79 |

As tabelas tiram 13% da KL do nível 1 e 10% do nível 2, sem custo de RAM e com o nível 0 igual ao Q4_K_M. O nível 1 ainda perde para o Q5_K_M sendo maior. O número que decide é o do 8B ([notebook](../tests/kernel/strata_8b_subcelulas_colab.ipynb), A100, com a régua da seção 28).

**Velocidade** (container, Xeon com AVX-512 e AMX, 4 núcleos, `llama-bench --repack 0`, tokens/s):

| | geração, 1 thread | geração, 4 threads | prompt 512, 4 threads |
|---|---:|---:|---:|
| Q5_K_M do llama.cpp | 7,75 | 26,65 | 502 |
| Q5_K_M com prefetch (teste local) | 13,71 | 44,18 | — |
| nível 1 P | 13,75 | 43,64 | 474 |
| nível 1 D | 10,44 | 35,79 | 322 |
| Q6_K do llama.cpp | 10,22 | 32,19 | 460 |
| Q6_K com prefetch (teste local) | 14,09 | 43,21 | — |
| nível 2 P | 12,83 | 40,91 | 477 |
| nível 2 D | 10,31 | 35,17 | 310 |

- **A geração dos tipos D estava 2× mais lenta que a dos P por falta de prefetch.** Com os pesos fora do cache, o prefetch do hardware não acompanha um kernel que faz mais conta por byte: as 112 matrizes de uma vez levavam 140 ms no D1 contra 61 ms no P1, e 74 contra 42 ms com cada matriz no cache. Pedir o bloco 8 à frente (`_mm_prefetch`) em todos os kernels R, P e D levou o D1 de 5,4 para 10,4 tokens/s numa thread e o P1 de 10,0 para 13,8. Os valores não mudam (KL igual).
- **Os k-quants do llama.cpp têm o mesmo problema nesta máquina.** O mesmo prefetch nos `vec_dot` do Q4_K, Q5_K e Q6_K do ggml (teste local, fora do patch) leva o Q5_K_M de 7,75 para 13,71 tokens/s numa thread. A comparação justa é com essa linha: aí os tipos P empatam ou quase (0,91–1,00) e os D ficam em 0,73–0,81 na geração, pelo segundo `maddubs` a cada 32 pesos. Contra o llama.cpp como ele é, o nível 1 D gera 1,35× mais rápido que o Q5_K_M; isso é do kernel do k-quant, não do formato, e só foi medido nesta máquina.
- **Sem `--repack 0`, os k-quants vão para o AMX**, que na geração é mais lento que o `vec_dot` aqui (Q5_K_M 7,81 tokens/s numa thread com AMX). Os tipos do STRATA não usam o AMX; a matriz de saída (Q6_K) usa, e isso pesava nos dois lados.
- **Prompt:** os tipos D fazem duas passadas no caminho tiled (códigos e δ), 0,64–0,67 do k-quant de mesma qualidade (antes das duas passadas, 0,12, porque caíam no `vec_dot`).

## 30. Sub-células por tensor no Llama-3.1-8B (tipos D na GPU)

Colab A100-80GB, [notebook](../tests/kernel/strata_8b_subcelulas_colab.ipynb) (commit `0cf4156`, ainda com o Q6_K_P1 de 274 bytes), KL contra o BF16 em 100 trechos do wikitext-2 raw de teste, imatrix de 125 trechos, tabela completa em [`resultados/unsloth__Meta-Llama-3.1-8B_subcelulas_a100.md`](../tests/kernel/resultados/unsloth__Meta-Llama-3.1-8B_subcelulas_a100.md). Conferência GPU contra CPU: tudo OK.

| | RAM (GiB) | KL ×1000 | k-quant F16 no mesmo tamanho | bits/peso a mais |
|---|---:|---:|---:|---:|
| nível 1 P | 5,512 (5,396 com o Q6_K_P1 compacto) | 10,78 | 7,13 | +0,34 |
| nível 1 D | 5,512 (5,396) | 9,27 | 7,13 | **+0,23** |
| nível 2 P | 6,208 | 3,66 | 3,09 | +0,20 |
| nível 2 D | 6,208 | 3,40 | 3,09 | **+0,11** |

(Q5_K_M com imatrix: 5,339 GiB, KL 7,55 em F16 e 8,60 no caminho real; Q6_K: 6,143 GiB, 3,31 e 3,47.)

- **O kernel dá o que a emulação previu:** 9,27 / 3,40 contra 9,24 / 3,39 da emulação "borda" (0,1 erro padrão). A seção 28 estimava +0,21 / +0,10 bit/peso; deu +0,23 / +0,11.
- **No mesmo tamanho, o STRATA ainda perde:** a KL do nível 1 fica 30% acima do k-quant com imatrix, a do nível 2 10% acima. As sub-células cortaram o atraso pela metade no nível 2 e em um terço no nível 1, sem mexer no nível 0 (igual ao Q4_K_M do hub). Contra o Q6_K do caminho real (3,47), o nível 2 D (3,40) parece ganhar; é a diferença de caminho da seção 28 (FP16 contra ativações em 8 bits), não do formato.
- **Geração na GPU reprovada:** os tipos D geram a 0,65 dos tipos P (94 contra 145 tokens/s); contra o k-quant, 0,64 no nível 1 e 0,79 no nível 2. A régua pedia 0,95. O kernel do MMVQ faz, a cada 4 pesos, dois `__vcmpeq4`, um `__byte_perm` duplo e um segundo `dp4a`; os tipos P já ficavam perto do limite da memória (145 contra 146 do Q5_K_M), então o que se soma de conta aparece direto no tempo. O prompt (6613 contra 4505) é o cuBLAS contra o MMQ, como antes.
- **Leitura:** o resultado confirma a seção 28. O preço de aninhar numa grade uniforme ficou em ~0,1–0,2 bit/peso por nível no 8B, e quatro vertentes não o fecharam. Os tipos D valem como opção na CPU (0,73–0,81 da velocidade, seção 29); na GPU, os tipos P continuam o padrão até o kernel D ser refeito.
