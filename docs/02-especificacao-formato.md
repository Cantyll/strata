# Especificação do formato `.strata`

Este documento descreve o formato em dois níveis:

- **Parte A — `strata/0.2` como implementado.** É o contêiner que o código de [`src/strata-core.js`](../src/strata-core.js) escreve e lê hoje, e que o leitor de referência em Rust ([`leitor-rs/`](../leitor-rs)) também lê. Todo byte descrito aqui foi testado: ida e volta bit a bit, propriedade de prefixo e bordas FP16/FP32/FP8/inteiros (ver [`tests/`](../tests)). As regras que o leitor aplica a um arquivo hostil estão em A.12 e os hashes de integridade em A.13.
- **Parte B — extensões propostas.** São o desenho completo da pesquisa ([01-pesquisa.md](01-pesquisa.md)), ainda sem código: modo residual, codec por bloco, genealogia no próprio arquivo, hashes por bloco e codificador para GPU.

Convenções: números em little-endian; offsets em bytes; "peso" = um elemento de um tensor.

---

## Parte A — `strata/0.2` e `0.3` (implementado)

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
| `tensors` | lista | um item por tensor, na ordem original: `name`, `dtype`, `shape`, `n` (número de elementos, ou `null` quando o tensor é guardado cru), `bytes` (tamanho original) e, com integridade, `sha256` (A.13) |
| `streams` | lista | a tabela de streams (A.4), já na ordem em que aparecem no corpo |
| `fonte` | objeto ou `null` | `{"tipo": "safetensors", "header_b64": ..., "sha256": ...}`: os bytes exatos do cabeçalho do safetensors original, em base64, para a volta bit a bit; `sha256` (opcional) é o SHA-256 do `.safetensors` original inteiro |
| `pai` | objeto ou `null` | reservado para a genealogia (Parte B); `null` no protótipo |
| `integridade` | objeto (opcional) | `{"alg": "sha256"}`: o arquivo traz os hashes de A.13 |

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
| `h` | com integridade: SHA-256 dos `l` bytes gravados (A.13) |

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

Qualquer outro dtype (`I64`, `I32`, `U8`, `BOOL`, ou um que o formato não conhece, como `F4`) vira um único stream **`raw`** no estrato 0, com os bytes originais. Um tensor de ponto flutuante pequeno também pode ir cru: com a opção `cruAbaixo` do escritor, todo tensor float com menos elementos que o limiar vira um stream `raw` e leva `n = null` (o conversor de vários arquivos usa 65.536; o padrão do core é 0, desligado). Estratificar uma escala de alguns KB não poupa nada que importe, e em modelos MoE essas escalas multiplicariam a tabela de streams (B.9). O escritor recusa um tensor cujo tamanho em bytes não bate com `shape × dtype`, como o próprio safetensors faz.

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

O exemplo está sem os hashes de A.13 (`integridade: false` no escritor). Com eles, o cabeçalho ganha 64 caracteres por stream e por tensor; o mesmo modelo com hashes é o vetor [`tests/vetores/ok-exemplo.strata`](../tests/vetores/ok-exemplo.strata).

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
- **Hashes por stream, não por bloco.** Cada stream e cada tensor têm seu SHA-256 (A.13), mas um stream é um plano inteiro de um tensor: não dá para conferir um pedaço de um stream sem baixar o stream todo.
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

### A.12 Regras do leitor

O leitor trata todo arquivo como hostil. As regras abaixo valem para os dois leitores (JS e Rust) e são conferidas antes de qualquer byte do corpo ser decodificado; qualquer violação recusa o arquivo inteiro (`StrataError` no JS, `Erro` no Rust). Um leitor novo tem que passar nos vetores de [`tests/vetores/`](../tests/vetores).

**Contêiner**

1. Pelo menos 12 bytes; `MAGIC` correto; versão 1 ou 2.
2. `2 ≤ HL ≤ 100 MiB` (o teto do safetensors) e `12 + HL ≤` tamanho do arquivo.
3. Cabeçalho em UTF-8 válido, sem BOM, JSON válido, sem surrogate solto, sem número fora do alcance de um double, e nenhum valor a mais de 64 níveis da raiz. Chaves repetidas: vale a última.
4. Presentes `format`, `ordem`, `estratos`, `tensors`, `streams`, `fonte` e `pai`. `format` é `"strata/0.1"` com versão 1 e `"strata/0.2"` com versão 2; `ordem` é `"estrato-maior"`; `pai` é `null` ou objeto. Chaves desconhecidas no topo são ignoradas (é o espaço para extensões como `leitura`).

"Inteiro" abaixo é um número sem parte fracionária entre 0 e 2^53 − 1.

**Tensores**

5. Cada item tem exatamente as chaves `name`, `dtype`, `shape`, `n`, `bytes` (mais `sha256` com integridade, e nenhuma outra).
6. `name` é texto e não se repete. `dtype` casa com `[A-Z][A-Z0-9_]{0,31}`. `shape` é uma lista de inteiros cujo produto não passa de 2^53 − 1.
7. Se o tamanho do dtype é conhecido, `bytes = produto(shape) × tamanho`. Para os tipos de ponto flutuante de A.5, `n = produto(shape)` ou `null`; para os outros, `n = null`. Um tensor com `n = null` é cru, qualquer que seja o dtype: tem só o stream `raw` e não conta para `estratos`.
8. `estratos = M + 1`, com `M` o maior número de bits de mantissa entre os tensores de ponto flutuante (0 se não houver).
9. A soma de `bytes` não passa de `limiteSaida`, por padrão `max(1 GiB, 64 × tamanho do arquivo)`, ajustável por quem chama. É o que impede uma bomba: um stream rANS de um símbolo só custa zero bit por elemento, então só o cabeçalho limita a saída. Um leitor em fluxo (que decodifica enquanto o arquivo chega) não conhece o tamanho final: ele aplica a mesma conta sobre os bytes já recebidos, à soma de `r` dos streams que for decodificar e, no fim, à soma de `bytes`. Pode recusar mais cedo um arquivo de compressão extrema que a leitura do arquivo inteiro aceitaria; quem chama resolve isso passando `limiteSaida`.

**Streams**

10. Cada item tem exatamente `t`, `s`, `p`, `o`, `l`, `r`, `c` (mais `h` com integridade).
11. `t` aponta um tensor existente. Para um tensor de ponto flutuante, `p` é `sign` ou `exp` (estrato 0) ou `mj` com `1 ≤ j ≤ M` do tensor (estrato `j`, escrito sem zero à esquerda); para os outros, só `raw` (estrato 0). `s` tem que ser esse estrato.
12. A tabela está em ordem estritamente crescente de `(s, t, sign antes de exp)`. Junto com "a tabela tem exatamente `2 + M_t` streams por tensor de ponto flutuante e 1 por tensor cru", isso garante: nenhum stream repetido, nenhum faltando, e os prefixos de A.6 corretos.
13. `o` é a soma dos `l` anteriores: os streams são contíguos, sem buraco nem sobreposição, a partir de 0.
14. `r` é o tamanho cru esperado: `ceil(n/8)` para `sign` e planos, `n` para `exp`, `bytes` para `raw`.
15. `c` é `raw` (e então `l = r`), `deflate`, ou `rans` (só no `exp` e só na versão 2).
16. O arquivo não tem nada depois do último stream. Menos que isso é um prefixo: só os streams inteiramente dentro dele podem ser lidos.

**Fonte**

17. `fonte` é `null` ou tem exatamente `tipo` (`"safetensors"`), `header_b64` e, opcionalmente, `sha256`.
18. `header_b64` é base64 canônico (com enchimento, sem espaço). Os bytes são um cabeçalho de safetensors válido pelas regras do safetensors: começa com `{`, até 100 MiB, `__metadata__` só com textos, cada tensor com exatamente `dtype`, `shape` e `data_offsets`, tamanho batendo com `shape × dtype`, e os `data_offsets` contíguos a partir de 0.
19. Os tensores da fonte são os mesmos do `.strata`: mesmos nomes, dtype, shape e tamanho.

**Decodificação de cada stream**

20. Os bytes do stream estão dentro do que foi lido; com integridade, o SHA-256 deles é `h`, conferido **antes** de descomprimir.
21. `deflate`: produz exatamente `r` bytes, para assim que passaria disso e consome todos os `l` bytes. Recusa o que o zlib recusa: código de Huffman com excesso, código incompleto (fora o caso de um código só), distância antes do início da saída, bloco guardado com complemento errado, tipo de bloco 3.
22. `rans`: além de A.11, a tabela tem símbolos em ordem estritamente crescente que somam exatamente 2^15, tem pelo menos um símbolo quando `r > 0`, e o estado inicial fica em `[2^23, 2^31)`.
23. Forma canônica: os bits de enchimento no último byte de `sign` e dos planos são zero, e todo byte de `exp` cabe no campo de expoente do dtype (menor que 2^E).
24. Lido na profundidade total, o tensor tem o SHA-256 `sha256` (com integridade). Na volta para `.safetensors`, o arquivo inteiro tem o SHA-256 `fonte.sha256`, quando ele existe.

### A.13 Integridade

Três níveis de hash, todos SHA-256, todos opcionais no formato e todos gravados pelo escritor por padrão (`encode(..., { integridade: false })` desliga os dois primeiros):

| Hash | Onde | Cobre | Quando é conferido |
| --- | --- | --- | --- |
| por stream | `h` em cada stream | os bytes gravados daquele stream | antes de descomprimir, inclusive numa leitura parcial |
| por tensor | `sha256` em cada tensor | os bytes originais do tensor | quando o tensor é lido inteiro |
| do original | `fonte.sha256` | o `.safetensors` original | na volta (`toSafetensors`); é o mesmo número que o Hugging Face publica para o arquivo (o oid do Git LFS) |

**Id do arquivo.** O SHA-256 dos bytes `[0, 12 + HL)` (`MAGIC`, `HL` e cabeçalho). Como o cabeçalho carrega todos os hashes acima, esse id sozinho autentica o arquivo inteiro. Com `idEsperado`, o leitor confere o id antes de interpretar o JSON, então um cabeçalho adulterado nem chega ao parser.

**O que os hashes protegem e o que não protegem.** Dentro do arquivo, eles pegam corrupção e adulteração no caminho, e garantem que nenhum byte não conferido chega ao descompressor. Contra quem escreveu o arquivo eles não protegem nada: quem muda um peso também refaz os hashes. A proteção contra isso é a âncora vinda de fora: o id publicado por quem distribui (num índice multi-arquivo, num hub, numa assinatura) ou o `fonte.sha256` comparado com o SHA-256 do `.safetensors` que o Hugging Face já publica. O safetensors não tem hash nenhum dentro do arquivo; quem confere hoje é o Git LFS, e só o arquivo inteiro.

**Custo medido** no Llama-3.2-1B: +104 KB de cabeçalho para 1.314 streams e 146 tensores (0,0064% do arquivo, 0,0007 bit/peso).

### A.14 Modo residual (`strata/0.3`)

Implementado no leitor de referência em Rust (`leitor-rs/src/residual.rs`, escritor `strata-ref residual`); o leitor em JS da página não lê a 0.3. No modo residual, o estrato 0 de cada matriz é um k-quant do llama.cpp (Q4_K ou Q6_K, os bytes do GGUF sem mudança), os estratos 1 a K são níveis de refinamento de 1 bit por peso, e o estrato K + 1 (o **fecho**) completa o BF16 exato. Ler até o estrato 0 dá o GGUF original byte a byte; até o nível k, o centro de uma célula 2^k vezes menor (B.8); tudo, o `.safetensors` original byte a byte.

**Contêiner.** `MAGIC` termina em `0x03` e `format` é `"strata/0.3"`. Tudo de A.2 a A.13 continua valendo, com estes acréscimos:

| Campo | Onde | Significado |
| --- | --- | --- |
| `niveis` | topo (obrigatório na 0.3) | K, de 0 a 8 |
| `estratos` | topo | o maior entre `M + 1` dos tensores de ponto flutuante comuns e `K + 2` (se há tensor residual) |
| `residual` | tensor | `{"grade": "Q4_K" ou "Q6_K", "permuta": h ou null}`; só para `BF16` 2-D com colunas múltiplas de 256 e `n` = número de elementos |
| `extra` | tensor | `"gguf"`: tensor cru que não existe no safetensors (o cabeçalho do GGUF, `rope_freqs`); não entra na volta para `.safetensors` nem em `fonte` |
| `gguf` | topo (opcional) | `{"alinhamento": a, "partes": [...], "sha256": ...}`: como remontar o GGUF do estrato 0 |

**Streams de um tensor residual** (n pesos, B bytes por super-bloco de 256 pesos: 144 no Q4_K, 210 no Q6_K):

| `p` | Estrato | `r` | `c` | Conteúdo |
| --- | --- | --- | --- | --- |
| `q` | 0 | n/256 · B | `raw` ou `deflate` | os super-blocos do GGUF, na ordem do GGUF |
| `n1` … `nK` | j | n/8 | `raw` ou `deflate` | o plano do nível j |
| `f` | K + 1 | 2n | `fecho` | o fecho (abaixo); `r` é o tamanho da saída |

**Ordem dos pesos.** Tudo é feito na ordem do GGUF: a ordem do safetensors (linhas × colunas), com as linhas permutadas como o `convert_hf_to_gguf` faz em `q_proj` e `k_proj` quando `permuta = h` (h = número de cabeças): a linha `c·L + 2i + s` do GGUF é a linha `c·L + s·L/2 + i` do safetensors, com `L = linhas/h`, `0 ≤ i < L/2`, `s ∈ {0, 1}`. O leitor desfaz a permutação na saída.

**Grade de cada peso.** Do super-bloco saem, para cada um dos 256 pesos, um passo `a`, um mínimo `m` e um índice inteiro `q`, exatamente como o ggml desquantiza (`dequantize_row_q4_K` e `dequantize_row_q6_K`), com `d` e `dmin` em FP16 convertidos para float32:

- **Q4_K** (`d`, `dmin`, `scales[12]`, `qs[128]`): sub-bloco j = peso/32, com `sc`, `mn` de 6 bits de `get_scale_min_k4(j)`; `a = d·sc`, `m = dmin·mn`; `q` = nibble baixo de `qs[32c + l]` para o peso `64c + l` e alto para `64c + 32 + l` (c < 4, l < 32); q de 0 a 15.
- **Q6_K** (`ql[128]`, `qh[64]`, `scales[16]` com sinal, `d`): na metade h (128 pesos) e l < 32, os pesos `128h + l + 32k` (k < 4) têm `q` = 4 bits de `ql` (baixos de `ql[64h + l]`, baixos de `ql[64h + l + 32]`, altos de `ql[64h + l]`, altos de `ql[64h + l + 32]`) mais os bits `2k, 2k+1` de `qh[32h + l]` acima deles, menos 32 (q de −32 a 31); `a = d·scales[8h + l/16 + 2k]`, `m = 0`.

**Sub-células.** O nível k divide a célula de q em 2^k partes, contadas no sentido de q crescente (r de 0 a 2^k − 1). Com `t = 2^(k+1)·q + 2r + 1 − 2^k`, o centro e as bordas são

```math
v = a\cdot\big(t\cdot 2^{-(k+1)}\big) - m, \qquad \text{bordas } e_\pm = a\cdot\big((t\pm1)\cdot 2^{-(k+1)}\big) - m
```

em float32, uma operação por vez, arredondando ao par mais próximo e sem FMA (`t·2^-(k+1)` é exato). Com k = 0, `v = a·q − m`: o valor do próprio k-quant. A sub-célula vai de min(e₋, e₊) inclusive a max(e₋, e₊) exclusive; a primeira sub-célula de q mínimo é aberta para baixo em q (para −∞ se a > 0, +∞ se a < 0) e a última de q máximo para cima. Se `a = 0`, ou `a`, `m` ou uma borda não são finitos, a sub-célula é a reta inteira. Como `t·2^-(k+1)` é exato, as bordas do nível k são bordas do nível K, e as sub-células se encaixam de um nível para o outro.

**Planos.** O escritor escolhe, para cada peso, a sub-célula r do nível K que o contém (descendo nível a nível; um peso fora de todas, saturado pelo k-quant, fica na mais próxima). O plano do nível j guarda o bit `K − j` de r (o nível 1 é o mais significativo): peso i no bit `i & 7` (o menos significativo primeiro) do byte `i >> 3`. Em cada super-bloco, os 32 bytes de um plano são o que o tipo `Q4_K_R<k>` (B.8) põe depois dos 144 bytes do Q4_K.

**Leitura parcial.** Com d ≤ K estratos, o peso vale `v` no nível d com `r_d = r >> (K − d)`, arredondado para BF16 (ao par mais próximo; NaN vira NaN quieto). Com d = 0, é o k-quant. Sem o fecho, o hash do tensor não é conferido (como em A.7).

**Ordinal BF16.** `ord(u) = u + 0x8000` para u sem sinal, `0x7FFF − (u & 0x7FFF)` com sinal: crescente com o valor, −0 logo antes de +0, NaN fora de [0x7F, 0xFF80]. Os BF16 de uma sub-célula [lo, hi) são os ordinais de `prim(lo)` a `prim(hi) − 1` (0x7F e 0xFF80 nas pontas abertas), onde `prim(x)` é o menor ordinal com valor ≥ x: `0x7FFF` para ±0; para x > 0, `0x8000 + (bits >> 16)` mais 1 se os 16 bits baixos não são zero; para x < 0, `0x7FFF − ((bits & 0x7FFFFFFF) >> 16)`. Os 7 bits baixos do ordinal são a posição dentro da **oitava** (`ord >> 7`, 0 a 511: sinal e expoente).

**Stream `fecho`.**

```
2                    número de contextos C
C × (1 + 2 + 4k)     contexto (uint8), k (uint16), k × (oitava uint16, freq uint16): contextos e oitavas
                     crescentes, freqs > 0 somando 2^15 (a regra de A.11 sobre as contagens)
4 × ⌈n/65536⌉        tamanho em bytes de cada bloco
...                  blocos: cada um com 65.536 pesos (o último com o resto), estado inicial uint32 e corpo,
                     como o rANS de A.11 (2^15, 2^23, estado volta a 2^23 com todos os bytes gastos)
```

O contexto de um peso é o campo de expoente de `a` (bits 23 a 30). Para cada peso, na ordem do GGUF, com a sub-célula [lo, hi] do nível K:

1. **Candidatas:** as `c` oitavas da tabela do contexto entre `lo >> 7` e `hi >> 7`. A massa de cada uma é a freq dela vezes quantos ordinais dela caem em [lo, hi]; `CM_i` é a soma das massas das i primeiras e `total = CM_c`.
2. **Símbolo:** com `R = 2^15 − 1`, a candidata i ocupa `[cum_i, cum_{i+1})`, `cum_i = ⌊(R − c)·CM_i / total⌋ + i`; o escape ocupa o slot R. Sem candidata (contexto ausente ou faixa vazia), o escape ocupa os 2^15.
3. **Posição:** dentro da oitava, o índice j entre os `n_o` ordinais dela na faixa, uniforme: `[⌊j·2^15/n_o⌋, ⌊(j+1)·2^15/n_o⌋)`; nada se `n_o = 1`.
4. **Escape:** o peso fora da faixa (ou NaN) vai cru: o byte alto e o baixo do BF16, cada um como símbolo uniforme de 128 slots.

O decodificador acha o símbolo por busca binária em `cum_i`, então cada peso custa O(log c) com qualquer tabela.

**GGUF do estrato 0.** `gguf.partes` lista, na ordem do arquivo, `{"t": i, "como": ...}`: `"cabecalho"` (primeira, um tensor `extra` com os bytes do GGUF até o início dos dados, já alinhados), `"q"` (o stream `q` do tensor residual i), `"f32"` (o tensor cru BF16 i alargado para float32, como o conversor guarda as normas) e `"bruto"` (um tensor `extra`). O GGUF é o cabeçalho seguido de cada parte, com zeros antes dela até o tamanho dos dados ser múltiplo de `alinhamento`. Com `sha256`, o resultado tem que conferir. Tudo isso está no estrato 0, então `bytes_ate(0)` basta.

**Regras do leitor (acréscimos a A.12).** Versão 3 com `"strata/0.3"`; `niveis` inteiro de 0 a 8; `residual` só com as chaves `grade` e `permuta`, grade conhecida, `BF16` 2-D, colunas múltiplas de 256, `permuta` nula ou h ≥ 1 com linhas múltiplas de 2h; `extra` só `"gguf"` e só em tensor cru; um tensor residual tem exatamente os streams `q`, `n1`…`nK` e `f`, nos estratos e tamanhos crus da tabela acima, e `f` só com `c: "fecho"`; no fecho, tabela bem formada, soma dos blocos = tamanho do stream, cada bloco com estado inicial em [2^23, 2^31) e estado final 2^23 com todos os bytes gastos; `gguf.partes` começa pelo cabeçalho, cada parte aponta um tensor do tipo certo e não se repete; tensores `extra` exigem a seção `gguf`.

**No Llama-3.2-1B** (Q4_K_M do `llama-quantize` sem imatrix, K = 3; 1.236 M pesos em matrizes, embedding incluído): estrato 0 5,18 bits/peso (o GGUF de 807,7 MB sai de um prefixo de 801,9 MB), níveis 1 bit cada, fecho 2,95; arquivo de 1,72 GB, 11,13 bits/peso. Detalhes em [05-resultados.md](05-resultados.md#21-modo-residual-gravado-strata03), seção 21. Vetores públicos em [`tests/vetores/residual/`](../tests/vetores/residual/vetores.json).

---

## Parte B — extensões propostas (ainda sem código)

### B.1 Modo residual

Implementado na `strata/0.3` com estrato 0 Q4_K/Q6_K ([A.14](#a14-modo-residual-strata03)). O resto desta seção é o desenho geral.

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

### B.4 Integridade por bloco (árvore Merkle)

Implementado: um hash por stream e um por tensor, listados no cabeçalho, e o id do arquivo como raiz (A.13). Isso já é uma árvore de altura 2, e com uma folha por stream ela é pequena: 1.314 folhas no Llama-3.2-1B, cerca de 6.500 num modelo de 70 B. Uma árvore binária só passa a valer com o codec por bloco (B.2): com uma folha por bloco de 4.096 pesos e por plano, um modelo de 70 B teria cerca de 150 milhões de folhas (5 GB de hashes), que não cabem no cabeçalho. Aí a raiz fica no cabeçalho, as folhas viram um stream próprio e cada bloco baixado por HTTP Range é conferido com o caminho até a raiz (log₂ das folhas, uns 27 hashes). Folha e nó com prefixos diferentes (0x00 e 0x01, como no RFC 6962), para que uma folha não se passe por um nó.

### B.5 Codificador

Codificador de entropia **determinístico, só com inteiros** (rANS ou Huffman canônico), decodificável dentro da GPU como o DFloat11 e o ECF8 já fazem, com vetores de teste públicos para que o hash bata em qualquer máquina.

Medido sem mudar o formato ([05-resultados.md](05-resultados.md#18-segurança-do-leitor), "Expoente em blocos independentes"): dividir o expoente de cada tensor em blocos rANS de 64 K símbolos, cada um com estado próprio, a tabela do tensor e um offset de 4 bytes, custa 0,0009 bit/peso no Llama-3.2-1B e deixa a decodificação do embedding 3,9 vezes mais rápida em 4 núcleos. Proposta: um campo `bloco` no stream de expoente (símbolos por bloco) e a lista de offsets logo depois da tabela.

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

**Tipo novo para os níveis 1 a 3 (o kernel): `Q4_K_R<k>`.** Super-bloco de 256 pesos = o bloco Q4_K do estrato 0 byte a byte (144 bytes) seguido de k planos de 32 bytes, um bit por peso por nível. São 176, 208 e 240 bytes (5,5, 6,5 e 7,5 bits/peso). Subir um nível é acrescentar um plano no fim do bloco, sem tocar no que já está na memória. Com `r` = os k bits do peso e `t = 2^{k+1}·q_4 + 2r + 1 − 2^k`, o valor é `v = d·sc·t/2^{k+1} − dmin·mn`; `t` cabe em int8 até k = 2, então o `vec_dot` contra ativações Q8_K é inteiro por sub-bloco como nos k-quants. Referência escalar em C em [`kernel/`](../kernel/README.md), conferida bit a bit no Llama-3.2-1B por `tests/kernel/test_q4kr.py`. No llama.cpp falta: o `ggml_type`, versões AVX2/NEON e CUDA (com fallback para desquantizar e multiplicar com cuBLAS enquanto não houver `mmvq`). O conversor do STRATA escreve esse tipo direto do arquivo residual, sem requantizar.

**`Q6_K_R<k>` para as matrizes que o Q4_K_M guarda em Q6_K** (15% dos pesos; sem níveis nelas o nível 1 perde para o Q5_K_M, [seção 22](05-resultados.md)). Super-bloco = o bloco Q6_K byte a byte (210 bytes) seguido dos mesmos k planos de 32 bytes, com o peso i na ordem do Q6_K (sub-bloco de 16 pesos `j = i/16`): 242, 274 e 306 bytes. Com `a = d·sc_j` (escala int8, com sinal) e `q` de −32 a 31, `t = 2^{k+1}·q + 2r + 1 − 2^k` e `v = a·t/2^{k+1}`, sem mínimo. A sub-célula `r` conta no sentido de `q` crescente, que é o de valor decrescente quando `sc < 0` (metade dos sub-blocos do Llama-3.2-1B). No produto escalar a soma do bloco vai em 64 bits, porque com k = 3 o pior caso passa de 2^31.

**Blocos montados na carga (`Q4_K_P<k>`, `Q6_K_P<k>`, k = 1 ou 2): só na RAM, nunca no arquivo.** O arquivo guarda os tipos R (base + planos, na ordem do download); o leitor pode, ao carregar, converter cada bloco uma vez num bloco com o índice inteiro de cada peso junto, que dá exatamente os mesmos valores. `Q4_K_P1` (176 bytes) tem o arranjo do `block_q5_K` do ggml com u = 2·q4 + r; `Q4_K_P2` (208 bytes) guarda os 4 bits baixos de u = 4·q4 + r como o `qs` do Q4_K e os 2 altos num `qh[64]`; `Q6_K_P<k>` (274 bytes) guarda w = 2^k·q + r em int8, seguido das 16 escalas e de d. Em todos, `t = 2u + 1 − 2^k` (ou `2w + 1 − 2^k`) é o `t` do tipo R. Layout exato em `kernel/strata_q4kr.h`; motivo e medidas em `kernel/README.md` (o produto escalar do nível 1 empata com o Q5_K; o do nível 2 fica a 1,27× do Q6_K numa thread e empata com 4).

**O que o tipo novo precisa ter, medido no nível 1** ([05-resultados.md](05-resultados.md), seção 20). Gravado como `Q5_K`, o nível 1 perde a vantagem que tinha na emulação (ΔPPL +1,43% contra +1,17%) e fica atrás do Q5_K_M (+1,25%) com a mesma RAM.

- **O mínimo deslocado, sem requantizar.** Requantizar `dmin·mn + d·sc/4` em 6 bits sobe o erro dos pesos de 4,03% para 4,23%, e é a única diferença que sobra entre a emulação com os saturados cortados (+1,18%) e o arquivo `Q5_K` (+1,43%). Um `dmin` escolhido por busca em vez de `max/63` só recupera 0,01 ponto. O tipo novo guarda o mínimo do Q4_K como está e aplica o deslocamento `d·sc(1/2 − 1/2^{k+1})` na desquantização, que é exato e não custa bits.
- **Sem lista de pesos saturados.** Cortar os pesos que saturavam nas pontas da grade (0,14%) na sub-célula da ponta, como o `Q5_K` faz, muda o ΔPPL de +1,17% para +1,18% no nível 1 e de +0,308% para +0,313% no nível 2. Eles não precisam de tratamento no kernel, e o arquivo deve adiar o valor exato deles para o último nível: o nível 1 cai de 5,95 para 5,84 bits/peso e o nível 2 de 6,95 para 6,91, com o mesmo total sem perdas (11,24).

### B.9 Modelos em vários arquivos

Modelos grandes chegam em vários `.safetensors` (shards) mais um `model.safetensors.index.json`. O `.strata` 0.2 converte um arquivo por vez e guarda um só cabeçalho de origem (`fonte.header_b64`, A.3). Há dois jeitos de cortar um modelo em vários `.strata`, e os dois estão implementados em [`tests/test-shards.mjs`](../tests/test-shards.mjs) sobre os **mesmos streams** do core (mesmos bytes no corpo; só muda onde cada stream mora). Medidas na [seção 19 dos resultados](05-resultados.md#19-modelos-em-vários-arquivos-llama-31-8b-4-shards).

Os dois desenhos têm um índice `model.strata.index.json` com `format: "strata-shards/0.1"` e campo `desenho`. Os dois guardam os bytes do `model.safetensors.index.json` original (`fonte.index_b64`) e o SHA-256 de cada arquivo `.strata` e de cada shard de origem, então a volta reconstrói o repositório de pesos inteiro bit a bit.

**Desenho 1, `por-shard`.** Um `.strata` 0.2 por shard, byte a byte o que o conversor de um arquivo grava (o teste confere a igualdade com `S.encode`). Cada um se abre sozinho no core e na página. O índice tem `weight_map` (tensor → arquivo, como o do safetensors) e, por arquivo, `fonte`, `fonte_sha256`, `bytes`, `id` (o id do arquivo da A.13), `sha256` (do arquivo inteiro, para quem baixa), `cabecalho` e `prefixos` (`P_0 … P_M` de A.6). Os hashes por stream e por tensor ficam no cabeçalho de cada arquivo, como a A.13 define; o leitor confere o `id` de cada arquivo contra o índice antes de ler o JSON. Com os prefixos no índice, ler o modelo até o estrato `d` é uma requisição `Range: bytes=0-(P_d − 1)` por arquivo, sem uma ida antes para ler o cabeçalho.

**Desenho 2, `por-estrato`.** Um arquivo por estrato com aquele estrato de **todos** os tensores do modelo, na ordem dos shards e dos tensores, dividido em partes de até um teto (padrão 5 GB; um stream nunca é cortado). Nome: `model-eSS-PPPPP-of-QQQQQ.strata` (estrato `SS`, parte `PPPPP` de `QQQQQ`). Cada arquivo começa com 16 bytes:

```
offset  tamanho  campo
0       8        53 54 52 41 54 41 45 01   ("STRATAE", versão 1)
8       4        estrato (uint32)
12      4        parte, contando de 0 (uint32)
16      ...      streams, um após o outro
```

O índice carrega o que nos arquivos 0.2 fica no cabeçalho: `estratos`, `tensors` (cada um com `shard`, o índice do shard de origem), `arquivos` (`nome`, `s`, `parte`, `bytes`, `sha256`), a tabela de streams (`t`, `s`, `p`, `o` = offset absoluto no arquivo, `l`, `r`, `c`, `h` e `f` = índice em `arquivos`; em binário, num arquivo à parte, ver abaixo) e `fonte.shards` (`nome`, `bytes`, `sha256`, `header_b64` de cada shard). Com hashes, cada tensor leva `sha256` e o topo leva `integridade: {"alg": "sha256"}`, nas mesmas regras da A.13; o leitor confere o `h` de cada stream antes de descomprimir. Os arquivos de estrato não têm cabeçalho próprio, então a raiz da confiança é o SHA-256 do índice: publicado ou assinado, ele autentica cada byte de cada arquivo (índice → tabela de streams → hash de cada stream). Ler até o estrato `d` = baixar os arquivos `e00` a `eDD` inteiros. O leitor recusa um arquivo cujo preâmbulo não bate com o estrato e a parte que o índice diz.

**Escolha: desenho 2 para distribuir; desenho 1 continua como exportação.** Os dois têm o mesmo tamanho e a mesma volta bit a bit. A diferença é operacional:

- No desenho 2, escolher a precisão é escolher **arquivos**, o que as ferramentas de hoje já fazem (`hf download --include "model-e0[0-2]-*"`, cache do Hub, espelhos, torrents). No desenho 1 é preciso um cliente que faça requisições por faixa de bytes em cada shard.
- Subir de precisão depois é baixar mais arquivos, sem tocar nos que já estão no disco.
- O desenho 2 pode ser gravado em fluxo (cada stream vai para o fim do arquivo do seu estrato), sem a segunda cópia que a ordem estrato-maior exige dentro de um arquivo do desenho 1. (O script de teste ainda monta os dois a partir das mesmas peças, para comparar os mesmos bytes.)
- Custo: nenhum arquivo de estrato se lê sozinho. O índice vira a peça crítica e precisa de integridade forte (B.4): ele é o que deve ser assinado.

**Limites do Hugging Face** (docs de 2026): 500 GB por arquivo no máximo, recomendação de menos de 200 GB, menos de 10 mil arquivos por pasta e menos de 100 mil por repositório. Um `.strata` único sem perdas (≈ 10,6 bits/peso) fica abaixo de 500 GB até ≈ 378 bilhões de pesos BF16; acima disso, um arquivo só não é possível.

**Tabela de streams em modelos MoE.** Em modelos com dezenas de milhares de tensores, a tabela por tensor e por plano cresce: o DeepSeek-V3 tem 91.991 tensores, 45.867 deles escalas F32 pequenas (0,006% dos pesos) que virariam 25 streams cada, e a tabela teria 1,38 milhão de entradas (≈ 100 MB em JSON). Duas regras resolvem antes de virar problema: tensores abaixo de um limiar (por exemplo 64 Ki elementos) vão crus num stream só, e a tabela passa a binária. Com os hashes, a tabela inteira do DeepSeek-V3 em JSON (≈ 205 MB) passaria do teto de 100 MB de cabeçalho da A.12; no D1 ela se divide em 163 cabeçalhos de ≈ 1,3 MB, no D2 fica toda no índice.

As duas regras estão implementadas:

- **Tensores pequenos crus** (A.5, opção `cruAbaixo`, 65.536 elementos no conversor de vários arquivos). Vale nos dois desenhos, porque muda os streams. Os dois leitores (JS e Rust) aceitam `n = null` num dtype de ponto flutuante como tensor cru; há um vetor público válido (`ok-cru-pequenos.strata`) e um hostil (`ruim-cru-com-planos.strata`).
- **Tabela binária no D2.** O índice deixa de ter `streams` e passa a ter `tabela: {nome, registros, bytes, sha256}`, apontando para o arquivo `model.strata.streams`. O SHA-256 da tabela fica no índice, então o hash do índice continua sendo a raiz. Formato:

```
offset  tamanho  campo
0       8        53 54 52 41 54 41 54 01   ("STRATAT", versão 1)
8       4        número de registros (uint32)
12      4        tamanho do registro (uint32): 36 sem hashes, 68 com integridade
16      ...      registros, na ordem (s, t, sign antes de exp)

registro:
0   4   t        índice do tensor em `tensors`
4   1   s        estrato
5   1   parte    0 raw, 1 sign, 2 exp, 2 + j = plano j
6   1   c        0 raw, 1 deflate, 2 rans
7   1   zero
8   4   f        índice do arquivo em `arquivos`
12  8   o        offset no arquivo
20  8   l        tamanho gravado
28  8   r        tamanho cru
36  32  h        SHA-256 do stream (só com integridade)
```

O leitor confere o SHA-256 da tabela contra o índice antes de ler qualquer registro, e depois cada registro: tensor e arquivo dentro das listas, parte coerente com o tipo do tensor, estrato igual ao do arquivo, faixa dentro do arquivo, rANS só no expoente e ordem estritamente crescente. O índice JSON tem o mesmo teto de 100 MB de um cabeçalho. Os cabeçalhos do D1 continuam em JSON, porque são o formato 0.2 de um arquivo só e ficam pequenos divididos por shard.
