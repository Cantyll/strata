# Fórmulas

Toda a matemática usada no STRATA, do porquê das camadas às contas que o protótipo faz. Cada fórmula aponta para onde ela vive no código e, quando há medida, para o número que a confirma. Os números são do modelo sintético BF16 de 1.000.192 pesos, semente 7 ([05-resultados.md](05-resultados.md)).

## 1. O número de ponto flutuante

Um peso guardado com `E` bits de expoente e `M` bits de mantissa tem três campos: sinal `s`, expoente `e` e mantissa `f`. Para números normais (`0 < e < 2^E − 1`):

```math
x = (-1)^{s} \cdot 2^{\,e-b} \cdot \left(1 + \frac{f}{2^{M}}\right), \qquad b = 2^{E-1} - 1
```

| Tipo | E | M | b (viés) |
| --- | --- | --- | --- |
| BF16 | 8 | 7 | 127 |
| FP16 | 5 | 10 | 15 |
| FP32 | 8 | 23 | 127 |
| FP8 E4M3 | 4 | 3 | 7 |
| FP8 E5M2 | 5 | 2 | 15 |

`e = 0` codifica zero e subnormais (`x = (−1)^s · 2^{1−b} · f/2^M`); `e = 2^E − 1` codifica infinito e NaN. O fator `q = 1 + f/2^M`, entre 1 e 2, é o **significando**.

Código: `LAYOUTS`, `makeToFloat` em `src/strata-core.js`.

## 2. Por que as camadas saem de graça: regra da cadeia

Seja `W` o conjunto de pesos e `S = g(W)` um esboço calculado deles (um arredondamento, uma quantização, ou "só o sinal e o expoente"). Como `S` é função de `W`, `H(S | W) = 0`, e então:

```math
H(W) = H(W, S) = H(S) + H(W \mid S)
```

Guardar primeiro o esboço e depois só o que falta para chegar a `W` custa, em teoria, exatamente o mesmo que guardar `W` direto. Encadeando esboços cada vez mais finos `S_0, S_1, …, S_M = W`:

```math
H(W) = H(S_0) + \sum_{j=1}^{M} H(S_j \mid S_0, \ldots, S_{j-1})
```

É o **refinamento sucessivo** (Equitz e Cover, 1991). No STRATA, `S_0` = sinal + expoente e cada `S_j` acrescenta um bit da mantissa.

## 3. Quanto custa cada estrato

Bits por peso para ler até o estrato `k`:

```math
R_k = H(s) + H(e) + \sum_{j=1}^{k} H(b_j \mid s, e, b_1, \ldots, b_{j-1}) \;\le\; 1 + H(e) + k
```

Entropia do expoente, a única parte realmente comprimível:

```math
H(e) = -\sum_{v} p(v) \log_2 p(v)
```

Medido: `H(e) = 2,70` bits (de 8 que o campo ocupa). O sinal custa ~1 bit e cada bit de mantissa ~1 bit, porque é quase aleatório. Então, no limite de um codificador de entropia ideal:

```math
R_k \approx 3{,}70 + k \quad \text{bits/peso}
```

Com o deflate do navegador o protótipo mede `R_k = 4,18 + k` (o deflate perde ~0,48 bit/peso no expoente). Código: `byteEntropy`, `encode` em `src/strata-core.js`.

## 4. O teto sem perdas

Para BF16 (16 bits por peso), o menor tamanho possível sem perder nada é aproximadamente:

```math
\frac{R_M}{16} \approx \frac{1 + H(e) + 7}{16} = \frac{10{,}70}{16} \approx 66{,}9\%
```

Medido com deflate: `11,16 / 16 = 69,8%` dos pesos, **70,0%** do arquivo inteiro (com o cabeçalho). É por isso que nenhum formato sem perdas tira muito mais de ~30% de um modelo BF16 isolado: os 7 bits de mantissa são quase ruído.

## 5. Prefixo do arquivo

Com os streams gravados em ordem estrato-maior, o número de bytes para ler até o estrato `d` é:

```math
P_d = 12 + HL + \sum_{e \,:\, e.s \le d} \ell_e
```

`HL` = tamanho do cabeçalho JSON, `ℓ_e` = tamanho codificado do stream `e`, `e.s` = estrato dele. A fração do arquivo é `P_d / P_M`. Medido: E0 = 37,7%, E1 = 46,5%, E3 = 64,3%. Código: `encode` (`prefix`).

## 6. Reconstrução no ponto médio

Lendo `k` dos `M` bits de mantissa, sobram `D = M − k` bits desconhecidos. O leitor liga o bit `2^{D−1}`, o que põe o valor no meio do intervalo possível:

```math
\hat{u} = \big(u \;\&\; \lnot(2^{D}-1)\big) \;|\; 2^{D-1} \qquad (0 < e < 2^{E}-1)
```

O bit não é ligado para zero, subnormais, infinito e NaN, para que zero continue zero e infinito não vire NaN. Código: `truncMid`, `joinStrata`.

**Limite do erro por peso.** O significando verdadeiro está num intervalo de largura `2^{−k}`, e o ponto médio erra no máximo metade disso:

```math
\frac{|\hat{x} - x|}{|x|} \;\le\; \frac{2^{-(k+1)}}{q} \;\le\; 2^{-(k+1)}
```

## 7. Cada estrato divide o erro por dois

O erro relativo cai pela metade a cada bit lido, o equivalente aos clássicos 6 dB por bit:

```math
\varepsilon_k \approx C \cdot 2^{-k}, \qquad \text{SNR}_k = -20 \log_{10} \varepsilon_k \approx 12{,}7 + 6{,}02\,k \ \text{dB}
```

Nos últimos estratos o ponto médio cai num valor que o formato consegue representar, e o erro passa a ser discreto. Com `D` bits desconhecidos, o erro em ULPs é `2^{D−1} − r`, com `r` uniforme em `{0, …, 2^D − 1}`, então:

```math
\mathbb{E}\big[(\hat{u}-u)^2\big] = \frac{4^{D} + 2}{12}\ \text{ULP}^2
\qquad\Rightarrow\qquad
\frac{\varepsilon_{k+1}}{\varepsilon_k} = \sqrt{\frac{4^{D-1} + 2}{4^{D} + 2}}
```

A previsão bate com a medida a partir de E2 (nos primeiros estratos o erro depende do próprio valor e a razão fica um pouco abaixo de 0,5):

| Passo | D antes → depois | Razão prevista | Razão medida |
| --- | --- | --- | --- |
| E2 → E3 | 5 → 4 | 0,5014 | 2,665 / 5,400 = 0,4935 |
| E3 → E4 | 4 → 3 | 0,5058 | 1,347 / 2,665 = 0,5054 |
| E4 → E5 | 3 → 2 | 0,5222 | 0,706 / 1,347 = 0,5241 |
| E5 → E6 | 2 → 1 | 0,5774 | 0,405 / 0,706 = 0,5737 |

## 8. Métrica de erro usada

Erro relativo RMS de cada tensor `t`, depois média ponderada pelo número de pesos `n_t`:

```math
\varepsilon_t = \sqrt{\frac{\sum_i (\hat{x}_i - x_i)^2}{\sum_i x_i^2}}, \qquad
\varepsilon = \frac{\sum_t n_t\, \varepsilon_t}{\sum_t n_t}
```

A média por tensor evita que um tensor pequeno de valores grandes (a normalização, com pesos perto de 1,0) domine a conta. Pesos não finitos são ignorados. No modelo de teste a soma usa todos os pesos (orçamento de 2 M ≥ 1 M pesos). O desenho de demonstração fica fora da conta. Código: `depthMetrics`.

## 9. Referências de quantização por bloco

Quantização simétrica por bloco de 32 pesos, com escala guardada em 16 bits:

```math
d = \frac{\max_{i \in \text{bloco}} |x_i|}{L}, \qquad q_i = \operatorname{clamp}\!\left(\operatorname{round}\frac{x_i}{d},\, -L,\, L\right), \qquad \hat{x}_i = q_i\, d
```

| Referência | L | Bits por peso | Erro medido |
| --- | --- | --- | --- |
| INT4 por bloco de 32 | 7 | 4 + 16/32 = 4,5 | 14,48% |
| INT8 por bloco de 32 | 127 | 8 + 16/32 = 8,5 | 0,83% |
| STRATA E8M1 | — | 5,16 (deflate) · 4,68 (limite) | 10,95% |
| STRATA E8M3 | — | 7,16 (deflate) · 6,68 (limite) | 2,67% |

E8M1 erra menos que o INT4 por bloco pagando de 0,2 (limite) a 0,7 bit (deflate) a mais por peso. Código: `blockQuantErr`.

## 10. Genealogia

O filho é guardado como a diferença bit a bit para o pai:

```math
\Delta = u_{\text{filho}} \oplus u_{\text{pai}}
```

`Δ` é estratificado e comprimido como qualquer tensor; um tensor idêntico ao do pai vira `HERDA` (32 bytes de referência). Medido: o filho como diferença ocupa **24,7%** do filho guardado sozinho.

**Quanto do ajuste sobrevive.** Com `δ_i` = diferença real entre filho e pai, o protótipo compara duas aproximações pelo cosseno:

```math
\cos(\delta, \tilde{\delta}) = \frac{\sum_i \delta_i \tilde{\delta}_i}{\sqrt{\sum_i \delta_i^2 \,\sum_i \tilde{\delta}_i^2}}
```

- Truncar os dois modelos em E8M3 e subtrair: `δ̃_i = x̂_filho,i − x̂_pai,i`. Cosseno medido: 0,27–0,51.
- Δ de 1 bit por peso com escala por bloco de 64: `δ̃_i = sinal(δ_i) · média_bloco |δ|`. Cosseno medido: 0,71–0,93.

Por isso o STRATA estratifica o `Δ` (sinal + escala no estrato 0), e não o peso. Código: `lineage`.

## 11. O fine-tune simulado

- **LoRA de posto 8:** `ΔW = κ · B A`, com `B` de `linhas × 8` e `A` de `8 × colunas` gaussianos, e `κ` escolhido para que `RMS(ΔW) = 0,01 · desvio(W)`.
- **Ajuste completo:** `Δw_i = g_i · 0,004 · (|w_i| + 0,25 · desvio(W))`, com `g_i ~ N(0, 1)`.
- **Congelado:** cópia idêntica.

Código: `synthChild`.

## 12. O modelo sintético

Cinco matrizes no formato de uma camada de LLM (vocabulário 30%, atenção q e v 12% cada, MLP up e down 23% cada) mais uma normalização. Cada linha tem escala própria e há caudas e canais atípicos:

```math
\sigma_r = \sigma \, e^{0{,}35\, g_r}, \qquad x_{rc} = g_{rc}\, \sigma_r \, a_{rc}\, c_c,
\qquad
a_{rc} = \begin{cases} 6 & \text{com prob. } 0{,}004 \\ 1 & \text{senão} \end{cases}
\qquad
c_c = \begin{cases} 8 & \text{nas 2 colunas atípicas} \\ 1 & \text{senão} \end{cases}
```

`g` são sorteios gaussianos `N(0, 1)`; `σ` vai de 0,012 a 0,020 conforme o tensor; `a` cria as caudas pesadas e `c` os canais atípicos.

Normalização: `1 + 0,05 g`. Tudo arredondado para BF16 com arredondamento ao par. Código: `synthModel`, `f32ToBf16`.

## 13. O desenho de demonstração

A página guarda um desenho de 240×180 como um tensor BF16 extra, para que a precisão seja visível. Intensidade `I ∈ [0, 1]` vira peso:

```math
w = 0{,}01 + 0{,}03\, I + \mathcal{N}(0,\, 0{,}0004^2), \qquad
I' = \operatorname{clamp}\!\left(\frac{\hat{w} - 0{,}01}{0{,}03},\, 0,\, 1\right)
```

O intervalo `[0,01; 0,04]` cobre três oitavas de expoente, então só com o estrato 0 o desenho já tem três tons. A lupa conta os tons distintos num recorte de 30 linhas do vaso: 3, 5, 10, 17 e 31 com 1 a 5 demãos, e 188 com o arquivo inteiro. Código: `prepare`, `convert` em `src/strata-lab.js`; `tonesAt` na página.

## 14. Estimativas de escala

```math
\text{tamanho} \approx \frac{N_{\text{pesos}} \cdot R}{8}\ \text{bytes}
```

| Caso | N | R (bits/peso) | Tamanho |
| --- | --- | --- | --- |
| gemma-2-9b-it, um `.strata` com todas as precisões | 9,24 × 10⁹ | 10,7 (limite) a 11,2 (deflate) | 12,4–12,9 GB (a página mostra ≈ 13 GB) |
| 1 trilhão de pesos | 10¹² | 10,7 | 1,34 TB (contra 2,0 TB em BF16) |
| Modelo de teste | 1.000.192 | 11,16 | 1,40 MB (contra 2,00 MB) |

Comparação da página: os 29 arquivos do repositório somam 191 GB, ou 97 GB com a deduplicação do Hugging Face (Xet); a linha do `.strata` é estimativa e aparece hachurada.

## 15. A prova de volta

O protótipo calcula o SHA-256 do safetensors original e do reconstruído. A chance de dois arquivos diferentes terem o mesmo resumo é desprezível (2⁻²⁵⁶ por comparação), então 64 de 64 caracteres iguais é a prova de que o arquivo voltou bit a bit. A página ainda compara os bytes um a um.
