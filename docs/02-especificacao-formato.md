# Especificação do formato `.strata`

Este documento descreve o formato em dois níveis:

- **Parte A — `strata/0.2` como implementado.** É o contêiner que o código de [`src/strata-core.js`](../src/strata-core.js) escreve e lê hoje. Todo byte descrito aqui foi testado: ida e volta bit a bit, propriedade de prefixo e bordas FP16/FP32/FP8/inteiros (ver [`tests/`](../tests)).
- **Parte B — extensões propostas.** São o desenho completo da pesquisa ([01-pesquisa.md](01-pesquisa.md)), ainda sem código: modo residual, codec por bloco, genealogia no próprio arquivo, hashes por bloco e codificador para GPU.

Convenções: números em little-endian; offsets em bytes; "peso" = um elemento de um tensor.

---

## Parte A — `strata/0.2` (implementado)

### A.1 Ideia

Cada número de ponto flutuante é desmontado nos seus campos (sinal, expoente e cada bit da mantissa). Os campos viram fluxos ("streams") separados, comprimidos um a um, e gravados **em ordem estrato-maior**: primeiro tudo o que pertence ao estrato 0 de todos os tensores, depois tudo do estrato 1, e assim por diante. Consequência: qualquer prefixo do arquivo que termine numa fronteira de estrato já contém um modelo inteiro, só que com menos precisão.

### A.2 Layout do arquivo

```
offset        tamanho   campo
0             8         MAGIC = 53 54 52 41 54 41 00 02   ("STRATA", 0x00, versão 0x02)
8             4         HL = tamanho do cabeçalho JSON, uint32 little-endian
12            HL        cabeçalho JSON em UTF-8
12 + HL       ...       corpo: os streams codificados, um após o outro, na ordem da tabela
```

O corpo não tem separadores: cada stream é localizado pelo seu offset `o` (relativo ao início do corpo, `12 + HL`) e pelo seu tamanho codificado `l`.

Versões: o último byte de `MAGIC` é `0x02` desde a introdução do rANS. Um leitor da 0.1 recusa o arquivo pela assinatura, em vez de ler o expoente rANS como cru. O leitor atual aceita `0x01` e `0x02`; arquivos 0.1 só usam `deflate` e `raw`.

### A.3 Cabeçalho JSON

| Campo | Tipo | Significado |
| --- | --- | --- |
| `format` | string | `"strata/0.2"` (arquivos antigos: `"strata/0.1"`) |
| `ordem` | string | sempre `"estrato-maior"` |
| `estratos` | inteiro | número de estratos = `M + 1`, onde `M` é o maior número de bits de mantissa entre os tensores de ponto flutuante (BF16 → 7, então 8 estratos) |
| `tensors` | lista | um item por tensor, na ordem original: `name`, `dtype`, `shape`, `n` (número de elementos, ou `null` quando o tensor é guardado cru) e `bytes` (tamanho original) |
| `streams` | lista | a tabela de streams (A.4), já na ordem em que aparecem no corpo |
| `fonte` | objeto ou `null` | `{"tipo": "safetensors", "header_b64": ...}`: os bytes exatos do cabeçalho do safetensors original, em base64, para a volta bit a bit |
| `pai` | objeto ou `null` | reservado para a genealogia (Parte B); `null` no protótipo |

### A.4 Tabela de streams

Cada item de `streams` tem:

| Chave | Significado |
| --- | --- |
| `t` | índice do tensor em `tensors` |
| `s` | estrato ao qual o stream pertence |
| `p` | parte do número: `"sign"`, `"exp"`, `"m1"` … `"mM"`, ou `"raw"` |
| `o` | offset do stream no corpo |
| `l` | tamanho codificado (bytes gravados) |
| `r` | tamanho cru (antes da compressão) |
| `c` | codificação: `"rans"` (A.11), `"deflate"` (deflate cru, RFC 1951) ou `"raw"` (sem compressão) |

A tabela é ordenada por `(s, t)`: estrato primeiro, tensor depois. O codificador fica, em cada stream, com a menor entre a versão crua, a deflate e (só no expoente) a rANS. Um leitor recusa um valor de `c` que não conhece, em vez de tratar o stream como cru.

### A.5 Como cada tensor vira streams

Tipos de ponto flutuante suportados como "float" (com estratos):

| dtype | bits | expoente (E) | mantissa (M) |
| --- | --- | --- | --- |
| `BF16` | 16 | 8 | 7 |
| `F16` | 16 | 5 | 10 |
| `F32` | 32 | 8 | 23 |
| `F8_E4M3` | 8 | 4 | 3 |
| `F8_E5M2` | 8 | 5 | 2 |

Para um tensor float com `n` elementos, cujo padrão de bits é `u` (inteiro sem sinal da largura do tipo):

- **`sign`** (estrato 0): 1 bit por elemento, empacotado em `ceil(n/8)` bytes. O elemento `i` fica no byte `i >> 3`, bit `7 − (i & 7)` (o primeiro elemento é o bit mais alto do byte).
- **`exp`** (estrato 0): 1 byte por elemento com o campo de expoente, `(u >> M) & (2^E − 1)`.
- **`mj`** (estrato `j`, para `j = 1 … M`): o plano de bits `j` da mantissa, do mais para o menos significativo, ou seja o bit `M − j` de `u`, empacotado como o sinal.

Qualquer outro dtype (`I64`, `I32`, `U8`, `BOOL`…) ou tensor de tamanho incompatível vira um único stream **`raw`** no estrato 0, com os bytes originais.

### A.6 Prefixos e leitura parcial

O tamanho do prefixo necessário para ler até o estrato `d` é:

```math
P_d = 12 + HL + \sum_{e \,:\, e.s \le d} e.l
```

Um leitor que tem só `P_d` bytes decodifica todos os tensores com `k = min(d, M)` planos de mantissa. A função `decodeTensorAt` do codec recusa ler além do prefixo disponível ("estrato fora do prefixo lido"), o que é a prova executável da propriedade de prefixo.

### A.7 Reconstrução com `k` planos (ponto médio)

Com `k < M` planos lidos, o leitor monta `u'` com sinal, expoente e os `k` bits altos da mantissa, e **liga o bit logo abaixo do último bit lido**:

```math
u' = \text{sinal} \;|\; (\text{exp} \ll M) \;|\; \text{bits altos} \;|\; 2^{\,M-k-1}
```

Isso coloca o valor no **ponto médio** do intervalo de números que compartilham aqueles bits, o que reduz o erro pela metade em relação a simplesmente zerar o resto. O bit do ponto médio **não** é ligado quando o expoente é 0 (zero e subnormais) nem quando é o máximo (infinito e NaN), para nunca transformar um zero em não-zero ou um infinito em NaN. Com `k = M` a reconstrução é exata.

### A.8 Volta para `.safetensors`

1. Decodificar com profundidade total (`k = M`): sai o padrão de bits original de cada tensor.
2. Ler `fonte.header_b64` e recuperar os bytes exatos do cabeçalho original (incluindo espaços de alinhamento e `__metadata__`).
3. Montar o arquivo: 8 bytes com o tamanho do cabeçalho (uint64 little-endian), o cabeçalho original e os bytes de cada tensor no seu `data_offsets`.
4. Conferir o SHA-256 do resultado contra o do original. No protótipo, os dois batem em 64 de 64 caracteres.

### A.9 Exemplo completo

Um tensor BF16 `2×4` com os valores `1, −0,5, 0,199, 0, 3,14, −123,5, 0,00995, 0,000999`, mais um tensor `I64` com `[7, 42]`. O safetensors tem 208 bytes; o `.strata`, 1.069 (num arquivo minúsculo o cabeçalho domina; num modelo de 1 M pesos ele é 0,4% do total).

Primeiros 12 bytes:

```
53 54 52 41 54 41 00 02 09 04 00 00
S  T  R  A  T  A  \0 v2 HL = 0x0409 = 1033
```

Cabeçalho (resumido; `header_b64` encurtado):

```json
{
  "format": "strata/0.2",
  "ordem": "estrato-maior",
  "estratos": 8,
  "tensors": [
    { "name": "exemplo.weight", "dtype": "BF16", "shape": [2, 4], "n": 8, "bytes": 16 },
    { "name": "exemplo.ids", "dtype": "I64", "shape": [2], "n": null, "bytes": 16 }
  ],
  "streams": [
    { "t": 0, "s": 0, "p": "sign", "o": 0,  "l": 1, "r": 1,  "c": "raw" },
    { "t": 0, "s": 0, "p": "exp",  "o": 1,  "l": 8, "r": 8,  "c": "raw" },
    { "t": 1, "s": 0, "p": "raw",  "o": 9,  "l": 8, "r": 16, "c": "deflate" },
    { "t": 0, "s": 1, "p": "m1",   "o": 17, "l": 1, "r": 1,  "c": "raw" },
    { "t": 0, "s": 2, "p": "m2",   "o": 18, "l": 1, "r": 1,  "c": "raw" },
    "... m3 a m7, um byte cada ..."
  ],
  "fonte": { "tipo": "safetensors", "header_b64": "eyJfX21ldGFkYXRhX18i…" },
  "pai": null
}
```

Prefixos: `P_0 = 1062`, `P_1 = 1063`, …, `P_7 = 1069` bytes. O que um leitor obtém em cada profundidade:

| Lido até | Valores reconstruídos |
| --- | --- |
| E0 (sinal + expoente) | 1,5 · −0,75 · 0,1875 · 0 · 3 · −96 · 0,01172 · 0,001465 |
| E1 | 1,25 · −0,625 · 0,2188 · 0 · 3,5 · −112 · 0,009766 · 0,001221 |
| E2 | 1,125 · −0,5625 · 0,2031 · 0 · 3,25 · −120 · 0,01074 · 0,001099 |
| E7 (tudo) | 1 · −0,5 · 0,1992 · 0 · 3,141 · −123,5 · 0,009949 · 0,0009995 |

Repare: o zero continua zero em qualquer profundidade, e cada estrato aproxima o valor do original pela metade da distância anterior. A volta deste exemplo para safetensors confere SHA-256 idêntico.

### A.10 Limites conhecidos da v0.2

- **rANS só de ordem 0 e só no expoente.** Uma tabela de frequências por tensor, sem contexto. Os planos da mantissa continuam em deflate (ou crus). Medidas de quanto o contexto ainda renderia estão em [05-resultados.md](05-resultados.md).
- **Decodificação sequencial.** Um estado de rANS por stream: não há estados intercalados nem blocos, então não dá para decodificar um tensor em paralelo na GPU (Parte B.5).
- **Um stream por plano por tensor.** Não há blocos de tamanho fixo, então não há decodificação paralela por bloco nem acesso aleatório dentro de um tensor.
- **Sem hashes por bloco.** A integridade é conferida no arquivo inteiro (SHA-256 da volta), não por bloco.
- **Sem genealogia no arquivo.** O campo `pai` existe, mas o delta entre modelos é calculado e medido pela função `lineage` sem ser gravado num `.strata`.
- **Cabeçalho em JSON.** Simples de inspecionar; num modelo com milhares de tensores, um índice binário seria mais compacto.

### A.11 Stream `rans` (expoente)

rANS de ordem 0 com estado de 32 bits e saída em bytes, só com inteiros, para que o mesmo arquivo saia idêntico em qualquer máquina. Probabilidades em `2^15`. Código: `ransEncode`, `ransDecode` e `ransFreqs` em [`src/strata-core.js`](../src/strata-core.js).

```
offset    tamanho   campo
0         2         k = número de símbolos presentes (uint16)
2         3k        k entradas: símbolo (uint8), freq (uint16); símbolos em ordem crescente
2+3k      4         estado inicial do decodificador x (uint32)
6+3k      ...       corpo: bytes lidos em ordem durante a decodificação
```

- **Frequências.** `freq[s] = max(1, round(cont[s] · 2^15 / n))`; enquanto a soma não for `2^15`, tira 1 (ou põe 1) do símbolo onde isso custa menos bits (ou rende mais), varrendo os símbolos em ordem crescente e ficando com o primeiro empate.
- **Decodificação** de `n = r` símbolos, com `cum[s]` = soma das freq dos símbolos menores:

```
slot = x & (2^15 − 1);  s = símbolo com cum[s] ≤ slot < cum[s] + freq[s]
x = freq[s] · (x >> 15) + slot − cum[s]
enquanto x < 2^23:  x = (x << 8) | próximo byte
```

- **Conferência.** O codificador começa com `x = 2^23`. Ao fim, o decodificador tem que voltar a `x = 2^23` tendo gastado todos os bytes; se não voltar, o stream está corrompido ou cortado e a leitura falha.

---

## Parte B — extensões propostas (ainda sem código)

### B.1 Modo residual

Em vez de "sinal + expoente" como estrato 0, o estrato 0 é um **quantizador de ponta** (GPTQ, AWQ, MXFP4/NVFP4 ou o Q4_K_M de um GGUF existente). Os estratos seguintes guardam o **resíduo** entre o valor reconstruído e o valor real, e o último fecha o valor exato como distância em ULPs. Pela regra da cadeia ([03-formulas.md](03-formulas.md)), isso não custa bits a mais em teoria, e as vistas intermediárias ficam tão boas quanto as do melhor quantizador.

### B.2 Codec por bloco

Cada bloco de 4.096 pesos declara o modo mais barato:

| Modo | Quando | Custo aproximado |
| --- | --- | --- |
| `HERDA` | bloco idêntico ao do pai | ~32 bytes (hash) |
| `DELTA` | fine-tune do pai | só os estratos do Δ |
| `ENTROPIA` | modelo-base, sem pai | ~10,7 bits/peso em BF16 |
| `SEMENTE` | perfil leve, com perda | semente + coeficientes (estilo SeedLM) |
| `BRUTO` | inteiros ou dados já comprimidos | tamanho original |

Blocos de tamanho fixo em elementos, com offsets no índice, permitem decodificar em paralelo na GPU e buscar um bloco isolado.

### B.3 Genealogia no arquivo

- O cabeçalho aponta o pai por **hash de conteúdo** (`pai: {"sha256": ..., "nome": ...}`).
- O filho guarda `Δ = filho XOR pai` (bit a bit), **estratificado**: o estrato 0 do Δ é sinal + escala por bloco (o BitDelta vira caso particular), e um estrato final em XOR garante o filho exato.
- Por que estratificar o Δ e não o peso: num fine-tune, a informação nova mora nos bits menos significativos. Truncar a precisão do filho apaga o ajuste; truncar o Δ não. No protótipo, o cosseno entre o Δ verdadeiro e o Δ que sobra ao truncar o filho em E8M3 foi 0,27–0,51; com um Δ de 1 bit por peso, 0,71–0,93.
- Riscos da cadeia: o filho não serve sem o pai. Saídas: cache de bases por hash, um quadro-chave completo a cada N gerações e exportação autônoma.

### B.4 Integridade

Hash por bloco organizado em **árvore Merkle**, com a raiz no cabeçalho. Permite conferir um prefixo, um bloco baixado por HTTP Range ou um bloco herdado de outro arquivo, sem ler o resto.

### B.5 Codificador

Codificador de entropia **determinístico, só com inteiros** (rANS ou Huffman canônico), decodificável dentro da GPU como o DFloat11 e o ECF8 já fazem, com vetores de teste públicos para que o hash bata em qualquer máquina.

### B.6 Distribuição e compatibilidade

- Índice no início e ordem estrato-maior: requisições **HTTP Range** baixam só a precisão desejada; o modelo fica mais preciso enquanto baixa.
- Blocos endereçados por conteúdo deduplicam no hub entre arquivos e famílias.
- Um **adaptador de safetensors virtual** apresenta um `.strata` como se fosse um `.safetensors` para transformers, vLLM e llama.cpp, sem mudar o código de quem usa.
- **Nenhum código executável no arquivo**: o modo `SEMENTE` usa geradores de especificação fixa, como o safetensors fez para escapar do pickle.

### B.7 Lista de leitura por tensor

Campo opcional `leitura` no cabeçalho: perfis com nome, cada um com um estrato por tensor, escolhido por quem escreve o arquivo para minimizar o dano sob um orçamento de bits.

```json
"leitura": [
  { "nome": "4.80", "bits_por_peso": 4.80, "estratos": { "model.layers.0.self_attn.q_proj.weight": 2, "...": 1 } }
]
```

- **Não muda nenhum stream**: o leitor decodifica cada tensor com `k = estratos[nome]` planos, pela mesma regra de A.7. Um leitor que ignora o campo continua lendo prefixos uniformes.
- **Custo**: um inteiro por tensor por perfil (112 bytes por perfil no Llama-3.2-1B). Quem escreve calcula os estratos com o proxy de Fisher numa amostra de texto (cerca de 1 minuto de GPU) e uma mochila de múltipla escolha (`tests/ppl/strata_alloc.py`).
- **Ganho medido** ([05-resultados.md, seção 14](05-resultados.md#14-estrato-por-tensor)): de 0,03 a 0,08 bit/peso pela mesma qualidade e, sobretudo, qualquer tamanho entre os degraus de 1 bit dos estratos uniformes.
- **Limite**: um perfil não é um prefixo único do arquivo, porque os streams estão ordenados por estrato (A.6). Na leitura remota isso são requisições por faixa de bytes (uma por tensor e estrato); em leitura sequencial, o leitor baixa até o maior estrato pedido e descarta o resto.

### B.8 Execução dos níveis do modo residual

O nível `k` sobre um estrato 0 Q4_K é uma grade de passo `d·sc/2^k` com as escalas do Q4_K. O valor lido no centro da sub-célula é

```math
v = \frac{d\,sc}{2^k}\, q_k - \Big(dmin\cdot mn + d\,sc\,\big(\tfrac12 - \tfrac{1}{2^{k+1}}\big)\Big), \qquad q_k = 2^k q_4 + b_k \in [0, 2^{k+4})
```

com `b_k` os `k` bits de refinamento do peso. Pesos que saturavam nas pontas da grade do Q4_K chegam exatos no nível 1 e recebem o `q_k` mais próximo.

| Nível | Índice | Tipo GGUF que já roda | Conversão |
|---|---|---|---|
| 0 | 4 bits | `Q4_K` (os próprios bytes) | nenhuma |
| 1 | 5 bits | `Q5_K` | escala exata (`d/2`, mesmo `sc`); o mínimo é requantizado em 6 bits por sub-bloco (`tests/ppl/strata_to_gguf.py`) |
| 2 | 6 bits | nenhum: `Q6_K` não tem mínimo | precisa de tipo novo |
| 3 | 7 bits | `Q8_0` com requantização, ou tipo novo | |

**Tipo novo para os níveis 2 e 3 (o kernel).** Super-bloco de 256 pesos: `d` e `dmin` FP16, 12 bytes de escalas e mínimos de 6 bits (empacotados como no Q4_K/Q5_K), índices de `4 + k` bits. Para `k = 2` são 208 bytes (6,5 bits/peso, o tamanho do Q6_K), e a desquantização é a do Q5_K com índices de 6 bits desempacotados como no Q6_K. No llama.cpp isso é um `ggml_type` novo com: struct do bloco, `dequantize_row` e `vec_dot` de referência na CPU, versões AVX2/NEON, e desquantização em CUDA (com fallback para desquantizar e multiplicar com cuBLAS enquanto não houver `mmvq`). O conversor do STRATA escreve esse tipo direto do arquivo residual, sem requantizar.

**O que o tipo novo precisa ter, medido no nível 1** ([05-resultados.md](05-resultados.md), seção 20). Gravado como `Q5_K`, o nível 1 perde a vantagem que tinha na emulação (ΔPPL +1,43% contra +1,17%) e fica atrás do Q5_K_M (+1,25%) com a mesma RAM.

- **O mínimo deslocado, sem requantizar.** Requantizar `dmin·mn + d·sc/4` em 6 bits sobe o erro dos pesos de 4,03% para 4,23%, e é a única diferença que sobra entre a emulação com os saturados cortados (+1,18%) e o arquivo `Q5_K` (+1,43%). Um `dmin` escolhido por busca em vez de `max/63` só recupera 0,01 ponto. O tipo novo guarda o mínimo do Q4_K como está e aplica o deslocamento `d·sc(1/2 − 1/2^{k+1})` na desquantização, que é exato e não custa bits.
- **Sem lista de pesos saturados.** Cortar os pesos que saturavam nas pontas da grade (0,14%) na sub-célula da ponta, como o `Q5_K` faz, muda o ΔPPL de +1,17% para +1,18% no nível 1 e de +0,308% para +0,313% no nível 2. Eles não precisam de tratamento no kernel, e o arquivo deve adiar o valor exato deles para o último nível: o nível 1 cai de 5,95 para 5,84 bits/peso e o nível 2 de 6,95 para 6,91, com o mesmo total sem perdas (11,24).
