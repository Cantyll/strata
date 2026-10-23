# Precisão sob demanda: plano e aplicação

Plano escrito em 2026-10-23, depois da distribuição sobre o GGUF do Hub ([resultados, seção 31](05-resultados.md#31-distribuição-sobre-o-gguf-do-hub-llama-31-8b-instruct)) e do congelamento dos níveis. Pergunta: o arquivo em camadas pode servir para **rodar** o modelo, e não só para guardar e baixar? A ideia é a de um vídeo na internet que começa em 480p e sobe para 1080p sem parar: o modelo começa a responder só com o estrato 0 (o Q4_K_M de verdade) e ganha os níveis de cima quando eles chegam do disco ou da rede, e no meio do caminho cabe no espaço de memória que estiver livre.

Regra deste plano: **o Llama-3.2-1B só descarta ideias; conclusão de qualidade só no 8B** (o 1B prometeu demais quatro vezes, seções 24 a 30). As contas sem modelo estão em [`tests/demanda/conta_demanda.py`](../tests/demanda/conta_demanda.py) e a medida do 1B em [`tests/demanda/resultados/`](../tests/demanda/resultados/).

## 1. Onde estamos (o que pesa a favor e contra)

| fato | número | fonte |
|---|---|---|
| Um arquivo sem perdas que começa no Q4_K_M do Hub intacto | 11,13 bits/peso no 8B; −60% a −66% contra guardar BF16 + Q4 + Q5 + Q6 | seção 31 |
| Cada nível acima é 1 bit cru por peso, num arquivo à parte | 1,00 GB por nível no 8B | seção 31 |
| O nível 0 roda em qualquer llama.cpp | é o próprio GGUF, mesmo SHA-256 | B.10 |
| Os níveis 1 e 2 rodam no llama.cpp com o patch (CPU e CUDA) | tipos P (43–46) e D (47–50) | seções 23 e 29 |
| **No mesmo tamanho de RAM, os níveis perdem para os k-quants com imatrix** | +0,23 b (nível 1 D) e +0,11 b (nível 2 D) no 8B | seção 30 |
| Velocidade de geração | P: 0,91–1,00 do k-quant na CPU, empata na GPU; D: 0,73–0,81 na CPU, 0,65 dos P na GPU | seções 29 e 30 |
| Leitura em fluxo (decodifica enquanto baixa) | já existe no leitor Python e no Rust | seção 18 |

Resumo honesto: o STRATA ganha em **guardar e distribuir** e perde por pouco em **qualidade por byte de RAM**. A precisão sob demanda não muda essa segunda linha. Ela só vale se o que a pessoa ganha for outra coisa: **começar antes, trocar de tamanho sem baixar outro arquivo e sem recarregar, e um só download servir para qualquer máquina.**

## 2. Quatro modos de uso

### 2.1 "Começa já": subir de nível com o modelo rodando

O que a pessoa faz: abre o modelo. O que acontece: o llama.cpp carrega o Q4_K_M e responde; em segundo plano os arquivos `model-e01` e `model-e02` chegam (do disco ou da rede), e cada matriz troca de nível entre um token e o próximo, quando o plano dela termina de chegar.

Conta do 8B (arquivos reais da seção 31; `conta_demanda.py`, parte 1), segundos até cada nível partindo do zero:

| rede | começa (Q4_K_M) | nível 1 | nível 2 | BF16 exato | Q5_K_M inteiro | Q6_K inteiro |
|---:|---:|---:|---:|---:|---:|---:|
| 50 Mbit/s | 787 | 948 | 1.109 | 1.788 | 917 | 1.055 |
| 100 Mbit/s | 394 | 474 | 554 | 894 | 459 | 528 |
| 300 Mbit/s | 131 | 158 | 185 | 298 | 153 | 176 |

- **Do zero, o ganho é só começar antes:** 394 s contra 528 s do Q6_K a 100 Mbit/s (25% antes). O nível 2 completo chega 5% **depois** do Q6_K inteiro (554 contra 528 s) e com qualidade um pouco abaixo dele. Quem baixa do zero e só quer o melhor arquivo de ~6,5 GB faz melhor baixando o Q6_K.
- **Para quem já tem o Q4_K_M, o ganho é grande:** subir ao nível 2 custa 2,01 GB de download contra 6,60 GB do Q6_K (−70%), e o modelo não para de responder enquanto isso.

### 2.2 "Cabe no que tem": tamanho contínuo pela RAM livre

O que a pessoa faz: diz quanta memória o modelo pode usar (`--strata-ram 5.8G`) ou deixa no automático. O que acontece: cada matriz recebe o nível 0, 1 ou 2 conforme a sensibilidade dela (as de atenção primeiro, como a alocação por tensor da [seção 14](05-resultados.md#14-estrato-por-tensor) já mostrava), até encher o orçamento. Qualquer tamanho entre 4,58 e 6,21 GiB no 8B, sem baixar outro arquivo.

**Ponto cego que esta conta expõe:** o repositório de GGUF do Hub já tem sete degraus nessa faixa (`conta_demanda.py`, parte 2):

| arquivo | GiB | degrau |
|---|---:|---:|
| Q4_K_M | 4,583 | |
| Q4_K_L | 4,946 | +0,36 |
| Q5_K_S | 5,215 | +0,27 |
| Q5_K_M | 5,339 | +0,12 |
| Q5_K_L | 5,641 | +0,30 |
| Q6_K | 6,143 | **+0,50** |
| Q6_K_L | 6,380 | +0,24 |

O "contínuo" só ganha espaço nos vãos, e o maior tem 0,5 GiB. Contra o repositório inteiro, a granularidade não é argumento. O que nenhum degrau do Hub faz é **mudar de tamanho sem outro download de 5–7 GB e sem recarregar o modelo**. O experimento E5 mede se, além disso, a curva do STRATA chega a passar a escada nos vãos (hipótese: entre 5,7 e 6,1 GiB, onde o melhor k-quant que cabe é o Q5_K_L e uma mistura dos níveis 1 e 2 já tem metade das matrizes no nível 2).

### 2.3 "Respira": descer de nível quando a memória aperta

O caso mais concreto. Llama-3.1-8B, cache de contexto (KV) em F16: 32 camadas × 8 cabeças × 128 × 2 × 2 bytes = **128 KiB por token**, 1 GiB a cada 8 mil tokens. Numa máquina de 8 GB com uns 6 GiB livres para o modelo:

| momento | contexto | KV | sobra para os pesos | k-quant do Hub | STRATA |
|---|---:|---:|---:|---|---|
| começo da conversa | 2 k | 0,25 GiB | 5,75 GiB | Q5_K_L (escolhido antes, fixo) | mistura nível 1/2 com 5,75 GiB |
| documento longo colado | 8 k | 1,0 GiB | 5,0 GiB | o Q5_K_L não cabe mais: troca de arquivo (outro download) ou corta o contexto | solta planos das matrizes menos sensíveis até 5,0 GiB, sem recarregar |
| conversa nova | 2 k | 0,25 GiB | 5,75 GiB | | lê os planos de volta do disco |

Com os planos num buffer à parte (seção 3), descer é só soltar memória: instantâneo. Subir é ler do disco: 1 GB por nível no 8B, uns 0,5 s num SSD NVMe e 2–10 s num SSD SATA ou HD (estimativa, a medir no E3).

### 2.4 "Rascunho e conferência": decodificação especulativa com uma cópia só

A ideia: o nível 0 propõe γ tokens (barato), o nível 2 confere todos de uma vez (uma passada só), e a saída tem exatamente a distribuição do nível 2 (Leviathan et al., 2023). Os pesos do nível 0 estão dentro do nível 2, então não há segunda cópia na RAM, **desde que a base fique separada dos planos** (seção 3). Com os tipos P de hoje (código inteiro de 5 ou 6 bits por peso), o rascunho leria os mesmos bytes do nível 2 e não ganharia nada.

O teto (`conta_demanda.py`, parte 3): na geração, que é limitada pela memória, o token do nível 0 custa c ≈ 4,80/6,80 = 0,71 do token do nível 2. Mesmo com o rascunho sempre aceito, o ganho máximo é 1/c = 1,41×. Com a taxa de aceitação α realista:

| α | c = 0,71 | c = 0,80 |
|---:|---:|---:|
| 0,85 | 1,05× | 1,00× |
| 0,90 | 1,10× | 1,03× |
| 0,95 | 1,17× | 1,08× |

**Medida no 1B (E1, seção 5): α = 0,886, teto de 1,08×; com a velocidade medida no container, 0,92×. Cortada.** Para comparar, um modelo pequeno separado como rascunho (o Llama-3.2-1B para o 8B, c ≈ 0,15) daria ~1,7× pela mesma conta com um α suposto de 0,7 (não medido aqui), ao custo de mais 0,7 GiB. **A especulativa entre níveis fica abaixo disso por construção; ela só sobrevive se o E1 der α ≥ 0,95**, e mesmo assim como bônus do layout separado, não como motivo para construí-lo.

## 3. O que precisa existir (peças)

| peça | o que é | apoia-se em | falta |
|---|---|---|---|
| **Layout separado (tipos S)** | a matriz no nível k = o bloco Q4_K/Q6_K do GGUF (pode ficar mapeado direto do arquivo do Hub) + k planos de 32 bytes por bloco num buffer à parte. O kernel lê dois fluxos. | tipos R (B.8, mesma conta, mesmo `t`), kernels AVX2/CUDA dos P e D, prefetch da seção 29 | AVX2 feito no E2 (`kernel/strata_separado*`); falta CUDA e a ligação ao ggml (E3) |
| **Troca a quente** | entre dois `llama_decode`, o tensor passa de Q4_K para S1/S2 (ou volta) trocando o tipo e o ponteiro dos planos; o grafo é refeito a cada lote, então a troca vale no próximo | `build_lora_mm`, por onde passam as multiplicações de pesos no grafo do llama.cpp | feito no E3 para CPU: os planos vão no `op_params` do tensor, e `llama_strata_tensor` dá acesso a ele; falta GPU |
| **Fonte dos planos** | `model-e01`, `e02` da distribuição sobre GGUF (B.10), lidos por tensor com o hash do stream conferido | `leitor-rs/src/dist.rs`, leitura em fluxo (seção 18) | ler um tensor por vez na ordem de sensibilidade, não na ordem do arquivo |
| **Escolha por tensor** | mochila: ganho de KL por byte de cada (tensor, nível), orçamento em bytes | `tests/ppl/strata_alloc.py` (Fisher, seção 14), imatrix | ganhos medidos com os tipos D, não com a mantissa do 0.2 |
| **Interface** | `--strata-ram 5.8G` / `--strata-nivel auto`; um log de "matriz X subiu para o nível 2" | | |

Os tipos D (sub-células) entram nos tipos S do mesmo jeito: a tabela de 16 bytes por tensor não depende do arranjo dos bytes.

## 4. Experimentos, em ordem

Cada um tem critério de corte. Se um corte acontece, os que dependem dele não rodam.

| # | hipótese | onde | custo | métrica | segue se | corta se |
|---|---|---|---|---|---|---|
| E0 | os números de tamanho e rede sustentam algum caso de uso | container, sem modelo | feito | segundos e GiB da seção 2 | | |
| E1 | o nível 0 concorda o bastante com o nível 2 para a especulativa | container, 1B, llama.cpp com o patch | **feito: α = 0,886, cortada** | "mesmo 1º token" e KL do nível 0 contra os logits do nível 2; tok/s dos dois | α ≥ 0,95 e c ≤ 0,75 | qualquer um fora: a especulativa entre níveis sai do plano |
| E2 | um kernel de dois fluxos (base + planos) gera na velocidade dos tipos P | container, microbenchmark C (o `llama-bench` passa para o E3, que liga os tipos ao llama.cpp) | **feito: nível 1 separado passa; nível 2 só misto (seção 6)** | ns/peso frio e quente, tok/s com 1 e 4 threads | ≥ 0,95 do P (a régua de sempre) | < 0,95: troca a quente com montagem P em segundo plano (custa a cópia de um tensor por vez e perde o "descer instantâneo") |
| E3 | dá para trocar o nível de um tensor entre tokens sem travar a geração e sem mudar o resultado | container, 1B | **feito: bit a bit igual; pausa < 1 token só cedendo um núcleo (seção 7)** | pausa máxima entre tokens (p99) durante a subida; logits depois da subida iguais bit a bit aos do GGUF do nível 2 carregado do zero; tempo de subida do disco | pausa < 1 token e bit a bit igual | pausa ≥ 1 token: fazer a troca em lotes menores; diferença nos logits: erro, não segue |
| E4 | o "começa já" funciona com rede de verdade | container, 1B, rede limitada (50/100/300 Mbit/s simulados) | **feito: a subida acompanha a rede sem travar além do ruído, preparando na própria geração (seção 8)** | tempo até o 1º token; tempo até o nível 2; tok/s durante a subida | 1º token no tempo do Q4_K_M sozinho (±5%) | |
| E5 | **no 8B, a curva KL × RAM da mistura por tensor passa a escada do Hub em algum vão** | Colab A100 (ou CPU High-RAM), 8B, tipos D | ~3 h de máquina | KL ×1000 contra o BF16 (100 trechos) em orçamentos de 4,6 a 6,4 GiB a cada 0,1, contra os 7 GGUF do Hub da seção 2.2 no mesmo caminho de ativações | vence o melhor arquivo que cabe em ≥ 1 faixa de 0,2 GiB | perde em todos: o argumento vira só "um download e troca sem recarregar" (que continua válido, e é o que o vídeo e o README dizem) |
| E6 | o pacote inteiro roda num 8B de verdade | CPU High-RAM no Colab ou máquina do Cantyll | 1 dia | demo: conversa que começa no Q4_K_M do Hub, sobe até o nível 2 e desce quando o contexto cresce; gravação | funciona de ponta a ponta | |
| E7 | portão do caminho 3: vale voltar a fechar a diferença de qualidade dos níveis? | | | o E5 e o uso do E6 | E5 mostrou vitória em vão, ou E6 mostrou uso real | senão os níveis ficam congelados, como decidido em 2026-10-22 |

O 1B decide E1 a E4, que são de engenharia e de concordância entre níveis, não de qualidade contra k-quant. Qualidade contra k-quant (E5) só no 8B.

## 5. Resultado do E1 (Llama-3.2-1B, container)

Llama-3.2-1B, llama.cpp com o patch, 60 trechos de 512 tokens do wikitext-2, logits do **nível 2** como referência ([tabela e comandos](../tests/demanda/resultados/unsloth__Llama-3.2-1B_especulativa_container.md)):

| contra o nível 2 | KL ×1000 | mesmo 1º token |
|---|---:|---:|
| nível 0 (Q4_K_M) | 47,1 | **88,6%** |
| nível 1 | 13,8 | 93,9% |
| Q6_K | 12,3 | 94,2% |

- **Aceitação do rascunho: α = 0,886.** Com o custo pela memória (c = 0,71), o melhor ganho é **1,08×** (γ = 2).
- **Velocidade medida:** o nível 0 gerou a 6,47 / 18,71 tokens/s (1 / 4 threads) e o nível 2 a 6,32 / 19,57. O nível 0 não foi mais rápido porque o `vec_dot` do Q4_K do ggml não tem o prefetch dos tipos do STRATA (seção 29). Com esse c ≈ 1, a especulativa daria 0,92×: mais lenta que rodar o nível 2 sozinho.
- **Decisão: a especulativa entre níveis sai do plano.** Mesmo no melhor caso (prefetch nos dois lados e α igual no 8B) o ganho fica em ~8%, abaixo de um rascunho separado pequeno. Se o layout separado (E2) existir por outros motivos, o rascunho vem de graça e pode ser medido de novo; não é motivo para construí-lo.
- Leitura de lado: o nível 1 concorda com o nível 2 quase tanto quanto o Q6_K (93,9% contra 94,2%), coerente com a seção 23.

## 6. Resultado do E2 (microbenchmark, container)

Kernel AVX2 dos tipos S em [`kernel/strata_separado_avx2.c`](../kernel/strata_separado_avx2.c), medido contra os tipos P
com a matriz fora do cache, formas em rodízio, mediana de 15 a 21 repetições, 2 rodadas com 1 thread e 3 com 4 threads
([tabelas e saída completa](../tests/demanda/resultados/e2_kernel_separado_container.md)). Os resultados são iguais bit a
bit aos dos tipos P.

| velocidade S / P, matriz fora do cache | 1 thread | 4 threads |
|---|---:|---:|
| Q4_K, nível 1 | 1,07 | 1,02 |
| Q4_K, nível 2 | 1,08 | 1,00 |
| Q6_K, nível 1 | 0,98 | 0,97 |
| Q6_K, nível 2 | **0,74** | **0,75** |
| matrizes das camadas (~85% Q4_K no 1B e no 8B), nível 2 tudo separado | 1,00 | **0,92** |
| matrizes das camadas, nível 2 misto (Q6_K montado ao subir) | 1,07 | 1,00 |

- **O nível 1 passa no layout separado em todas as grades.** A base continua sendo o arquivo do Hub, mapeado do disco, e
  subir é converter um plano de 32 bytes por bloco para a ordem em que o kernel lê (0,04 ns por peso, ~0,3 s no 8B).
- **O nível 2 separado não passa nos tensores Q6_K, e não tem como passar.** O Q6_K_P2 guarda o peso pronto em 8 bits e
  é mais rápido que o próprio Q6_K; o separado precisa desmontar o Q6_K e juntar dois planos, então o teto dele fica
  abaixo do P2. Nas matrizes das camadas, o nível 2 tudo separado dá 0,92 com 4 threads.
- **Decisão: layout misto.** Tensores Q4_K ficam separados nos dois níveis; tensores Q6_K ficam separados no nível 1 e,
  ao subir para o nível 2, são montados em Q6_K_P2 em segundo plano e trocados (5,8 ns por peso numa thread, ~6 s para
  o ~1,0 bi de pesos Q6_K das camadas do 8B). Com isso o nível 2 roda a 1,00–1,07 do P. A RAM é a mesma (274 bytes por bloco nos
  dois arranjos), mas por um tensor de cada vez existem as duas cópias: no 8B, o maior Q6_K com níveis (`ffn_down`, 59 mi
  de pesos) pede ~63 MB a mais durante a montagem. Descer continua imediato: o tensor volta para S1 apontando para a base mapeada e solta
  o P2 (se o sistema tiver tirado a base da memória, ela volta do disco por demanda).
- **Limites:** é microbenchmark com pesos aleatórios, não tokens/s; a máquina é compartilhada e uma rodada de 4 threads
  variou 15–20% (o rodízio mantém a comparação justa dentro de cada rodada, não entre rodadas); a proporção do 8B é
  estimada pela regra do `llama-quantize`, não contada; `token_embd` e `output` não ganham níveis e ficam fora da conta. O tokens/s de verdade sai do E3, com os tipos dentro do llama.cpp.

## 7. Resultado do E3 (Llama-3.2-1B, container)

O modelo é carregado do Q4_K_M, mapeado do disco. As 112 matrizes das camadas sobem para o nível 1 e depois para o 2
com o layout misto do E2. Uma thread de fundo prepara cada matriz, e a geração aplica a troca antes do próximo token.
O programa é [`tests/demanda/strata_subir.cpp`](../tests/demanda/strata_subir.cpp), com os tipos S e
`llama_strata_tensor` no patch de `kernel/ggml`
([tabelas, comandos e CSV por token](../tests/demanda/resultados/e3_troca_quente_container.md)).

- **Resultado igual bit a bit, nos dois sentidos.** Medimos os logits de 128 posições depois de subir para 1 e 2 e de
  descer para 1 e 0. Eles são idênticos aos do GGUF de cada nível carregado do zero, em 3 rodadas (15 de 15).
- **Velocidade igual à dos tipos P no modelo inteiro.** Com 4 threads, a razão das medianas de tokens/s é 1,00 no
  nível 1 e 1,01 no nível 2 misto. É a confirmação, em tokens/s, do que o E2 mediu no microbenchmark.
- **A troca custa microssegundos** (no máximo 16 µs por token). Descer é instantâneo e volta exatamente ao Q4_K_M.
- **A pausa depende de onde a subida roda.** Se a thread de fundo disputa os 4 núcleos com a geração, um token chega a
  levar **8 a 14 tokens normais**. As threads do ggml se esperam a cada operação, e uma delas fora do núcleo segura
  todas; baixar a prioridade não resolve. Com a geração em 3 threads enquanto a subida corre, a pausa máxima fica em
  **0,61 a 0,97 token** (4 rodadas). Passa no critério, mas a folga é curta, e os ~20 tokens da subida saem ~40% mais
  devagar.
- **Decisão: segue para o E4, com a regra "a subida tem núcleo próprio".** Com 4 núcleos, a geração cede um enquanto
  sobe. (O E4 trocou essa regra por uma melhor: preparar na própria geração, seção 8.) O que falta medir é o 8B: a subida dura ~7× mais e cada operação é maior, e a pausa precisa ser medida de novo
  lá (E6).
- **Limites:** só CPU com AVX2; a ordem de subida é a do arquivo, não a de sensibilidade; o prompt longo dos tipos S usa
  o caminho genérico do ggml, não o de tiles (mais lento, não medido); os planos vieram do cache do sistema, e o tempo
  de disco e de rede fica para o E4.

## 8. Resultado do E4 (Llama-3.2-1B, container, rede simulada)

Usamos o mesmo programa do E3, com os planos chegando a 50, 100 ou 300 Mbit/s, na ordem dos arquivos de estrato (todo o
nível 1, depois o nível 2), enquanto o modelo gera a partir do Q4_K_M
([tabelas, controles e CSV por token](../tests/demanda/resultados/e4_rede_container.md)).

- **O primeiro token sai na hora do Q4_K_M sozinho**, por construção: os planos chegam depois do GGUF. No 1B a
  100 Mbit/s, o nível 2 completo sai aos 84 s, contra 82 s do Q6_K baixado direto (+3%), a mesma conta da seção 2.1.
- **A subida acompanha a rede.** O nível 1 fica completo quando chega o último byte, e o nível 2 de 0,4 a 0,9 s depois.
- **A thread de fundo do E3 não serve com a rede.** Os planos chegam aos poucos, cada matriz vira uma rajada de
  trabalho, e as rajadas travam a geração: p99 de 270–400 ms e tokens de 10 a 11 tokens normais. Liberar um núcleo
  só durante a rajada não resolve, porque ela começa no meio de um token.
- **Preparar na própria geração resolve.** Entre dois tokens a geração gasta até 4 ms convertendo o que chegou. Os
  planos entram direto como tipos S. O Q6_K entra no nível 2 como `STRATA_Q6_K_S2`, um tipo novo, bit a bit igual ao P2,
  e é montado em P2 com o tempo que sobra, 2 a 6 s depois. O p99 (84–98 ms) e o pior token (1,9–2,4 normais) ficaram
  dentro do ruído da máquina, medido numa geração sem subida (p99 70–95 ms, pior token 1,7–4,3 normais).
- **Uma condição para isso: reservar a memória dos níveis antes de gerar.** Numa VM, converter em memória que acabou
  de ser alocada foi 10 a 20× mais lento por causa das faltas de página. Tocar os buffers antes custou 0,3 s para
  396 MB no 1B. Na prática, é escolher o nível-alvo (e a RAM) na partida.
- **"Pausa < 1 token" não é mensurável nesta máquina:** o próprio controle teve um token de 4,3 tokens normais. A
  medida fina vai para o E6, numa máquina estável.
- **Limites:** é rede simulada (sem perda, taxa fixa, sem conferir os hashes nem ler o formato `model-e0k`); o 1B só
  tem 4 núcleos e é ruidoso; no 8B o polimento do Q6_K leva ~7× mais tempo.

## 9. Riscos e pontos cegos

- **A qualidade por byte não muda.** Em qualquer orçamento que coincida com um degrau do Hub, a pessoa roda um pouco pior que o k-quant com imatrix daquele tamanho (+0,11 a +0,23 b de atraso no 8B). A precisão sob demanda troca isso por flexibilidade; precisa ser dito assim.
- **Só funciona no llama.cpp com o patch.** Ollama, LM Studio e o llama.cpp oficial não têm os tipos. Até isso mudar, quem usa é quem compila. O nível 0 continua rodando em todo lugar, e é por isso que a distribuição (seção 31) vale sozinha.
- **Quem é a pessoa que tem o problema?** "Já tenho o Q4_K_M e quero melhor sem baixar 6,6 GB" e "minha RAM muda durante a conversa" são suposições. O E6 é a forma barata de mostrar para alguém e ver se pedem.
- **Troca a quente mexe no coração do llama.cpp** (alocação dos buffers, mmap, grafo). Na CPU ficou pequena (E3: tipo, ponteiros e passos do tensor, sem mexer no grafo), mas depende de o llama.cpp não reempacotar os pesos e de ninguém mais segurar o ponteiro antigo. Na GPU é outro trabalho: a base precisa estar na VRAM e os tipos S não têm kernel CUDA.
- **A especulativa entre níveis tem teto baixo** (1,41× com aceitação perfeita). Não vender como ganho de velocidade.
- **No 8B, o KV e os buffers de computação também ocupam RAM;** o orçamento dos pesos é o que sobra, e o automático precisa medir isso em vez de supor.

## 10. Custo total e onde roda

| etapa | onde | quem roda | custo |
|---|---|---|---|
| E0, E1 | container | Claude | feito nesta sessão |
| E2, E3, E4 | container (4 núcleos, AVX-512, sem GPU) | Claude | 1 a 1,5 semana de trabalho de engenharia |
| E5 | Colab Pro, A100 High-RAM | Cantyll roda o notebook | ~3 h de máquina; o notebook segue o da seção 30 |
| E6 | Colab CPU High-RAM ou máquina local | Cantyll | 1 h |

## 11. Ordem de decisão

1. E1 (feito): a especulativa entre níveis saiu do plano (1,08× no melhor caso).
2. E2 (feito): layout misto. Separado para Q4_K nos dois níveis e Q6_K no nível 1; Q6_K no nível 2 montado em segundo plano.
3. E3 e E4 (feitos): a troca funciona com resultado bit a bit igual, acompanha a rede e, preparada na própria geração,
   não trava além do ruído da máquina.
4. E5 no 8B: existe vitória de qualidade em algum vão? Define o que se pode afirmar.
5. E6: demonstração de ponta a ponta.
6. E7: só então decidir se vale reabrir a qualidade dos níveis.
