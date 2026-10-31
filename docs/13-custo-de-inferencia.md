# 13 · Custo de inferência: o que está medido, o que se cruza e o que ninguém mediu

Pesquisa de literatura · 31 de outubro de 2026 · thread "Mapear lacunas no custo de inferência".

Este documento junta o que artigos, relatórios técnicos e repositórios já mediram sobre o custo de rodar
um LLM (memória, banda, KV cache, lote, energia, preço por token), cruza esses números entre si e com o
que o STRATA já mediu, e termina com uma lista de **lacunas**: combinações que ninguém mediu ou resolveu.

Partes vizinhas da mesma pesquisa, em threads próprias: quantização abaixo de 4 bits e codecs de pesos
(`docs/12`), motores de inferência local do GitHub (`docs/14`) e MoE, offload e paginação de pesos. Aqui fica
o fundamento de custo e o cruzamento.

**Como ler.** Cada número tem uma etiqueta:

- **[fato]**: número publicado, com link. Quando a fonte é secundária (blog, resumo, notícia), está dito.
- **[conta]**: aritmética minha sobre números publicados ou especificações. Dá para refazer.
- **[hipótese]**: inferência minha, não medida por ninguém. É o material das lacunas.

---

## 1. Resumo em seis frases

1. Gerar texto um token por vez é limitado pela **banda de memória**, não pela conta: para cada token, o
   modelo inteiro (ou a parte ativa, num MoE) atravessa o barramento. A conta de bolso é
   `tokens/s ≤ banda ÷ bytes lidos por token`.
2. Esse gargalo piora com o tempo: o poder de conta dos chips cresce ~3× a cada 2 anos e a banda da DRAM
   ~1,6× ([Gholami et al., 2024](https://arxiv.org/abs/2403.14123)). Cortar bytes é a alavanca que dura.
3. Em casa o lote é 1 (um usuário). Quase toda a literatura de custo (vLLM, DistServe, Splitwise) ganha
   juntando muitos usuários num lote; em casa a única forma de "encher o lote" é a **decodificação
   especulativa**, e ela rende mais justamente quando o modelo não cabe na VRAM (10 a 18×, SpecExec).
4. Compressão **sem perdas** já acelera a decodificação em lote 1, mas só em BF16 e FP8 (NWC 1,22× numa
   RTX 4070). No formato que as máquinas populares rodam (4 bits, Q4_K), o STRATA mediu que os códigos já
   estão quase sem redundância (3,86 de 4 bits): o teto ali é ~3%.
5. A máquina popular de 2026 tem 8 ou 16 GB de VRAM (54% dos PCs do Steam) e 16 ou 32 GB de RAM (80%); com
   isso, modelos densos de ~14B cabem em 4 bits na GPU e um 70B só roda dividido com a RAM, a ~2–3 tokens/s
   no teto teórico.
6. A lacuna mais concreta para o STRATA: **especulativa com o rascunho na VRAM (o Q4_K_M que já é o estrato
   0) e a conferência transmitindo só os estratos que faltam** pela PCIe, em vez do modelo inteiro. Os
   trabalhos publicados (SpecExec, SubSpec) transmitem a camada inteira; ninguém transmitiu só o resíduo.

---

## 2. Por que gerar texto custa banda, não conta

**Analogia.** Pense num cozinheiro rápido (a GPU ou a CPU) que, para cada prato (token), precisa buscar
todos os ingredientes na despensa (a memória). Se a despensa fica longe e o corredor é estreito, o
cozinheiro passa a maior parte do tempo esperando. Cozinhar mais rápido não ajuda; encurtar o corredor
(mais banda) ou levar menos ingredientes por prato (menos bytes por peso) ajuda.

### 2.1 O que está medido

- **[fato]** No modelo roofline, as camadas de projeção na fase de decodificação têm intensidade
  aritmética ~1 (uma operação por byte lido), bem abaixo do ponto em que o chip deixa de esperar a
  memória; o pré-preenchimento (ler o prompt) é limitado por conta
  ([Yuan et al., "LLM Inference Unveiled", 2024](https://arxiv.org/abs/2402.16363)).
- **[fato]** Pico de FLOPS de servidores cresceu 3,0× a cada 2 anos em 20 anos; banda de DRAM 1,6× e
  interconexão 1,4× no mesmo período ([Gholami et al., "AI and Memory Wall", 2024](https://arxiv.org/abs/2403.14123);
  número confirmado no resumo reproduzido pela [SemiEngineering](https://semiengineering.com/designing-ai-hardware-to-deal-with-increasingly-challenging-memory-wall-uc-berkeley)).
- **[fato]** A Databricks define **MBU** (utilização da banda do modelo) = banda atingida ÷ banda de pico,
  com banda atingida = (bytes dos pesos + KV cache) ÷ tempo por token. Mediram ~55% de MBU em 4× A100 e
  ~60% em 2× H100 com lote 1; o MBU cai quando o lote cresce
  ([Databricks, "LLM Inference Performance Engineering"](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)).
- **[fato]** Mesmo com lote grande, a decodificação continua limitada pela memória: os kernels de atenção
  e de multiplicação de matrizes ficam no regime de memória em todos os lotes testados, e mais da metade
  dos ciclos da atenção esperam a DRAM
  ([Recasens et al., "Mind the Memory Gap", 2025](https://arxiv.org/abs/2503.08311)).
- **[fato]** Energia: ler um operando de 32 bits da DRAM externa custa mais de 500 pJ, contra menos de 4 pJ
  de uma multiplicação de ponto flutuante de 32 bits, em 45 nm (Horowitz, ISSCC 2014, via
  [New Electronics](https://ne.mydigitalpublication.co.uk/june-2025/page-20); fonte secundária, não
  conferi a tabela original). Mover o byte custa ~100× mais que fazer a conta com ele.

### 2.2 A conta de bolso conferida com medidas reais

A tabela de desempenho do llama.cpp em chips Apple
([discussão #4167](https://github.com/ggml-org/llama.cpp/discussions/4167); Llama-2-7B, `llama-bench`,
geração de 128 tokens) permite conferir a conta `banda ÷ bytes`. **[conta]** Usei 6,74 B de pesos a 16,
8,5 e 4,5 bits/peso (F16, Q8_0, Q4_0).

| Chip (núcleos GPU) | Banda (GB/s) | F16 medido / teto | Q8_0 medido / teto | Q4_0 medido / teto |
|---|---:|---:|---:|---:|
| M1 (8) | 68 | – | 7,9 / 9,5 = **83%** | 14,2 / 17,9 = **79%** |
| M1 Pro (16) | 200 | 12,8 / 14,8 = **86%** | 22,3 / 27,9 = **80%** | 36,4 / 52,8 = **69%** |
| M1 Max (32) | 400 | 23,0 / 29,7 = **78%** | 40,2 / 55,9 = **72%** | 61,2 / 105,5 = **58%** |
| M3 Max (40) | 400 | 25,1 / 29,7 = **85%** | 42,8 / 55,9 = **77%** | 66,3 / 105,5 = **63%** |
| M4 Max (40) | 546 | 31,6 / 40,5 = **78%** | 54,0 / 76,2 = **71%** | 83,1 / 144,0 = **58%** |
| M2 Ultra (76) | 800 | 41,0 / 59,3 = **69%** | 66,6 / 111,7 = **60%** | 94,3 / 211,0 = **45%** |

O que a tabela ensina **[conta]**:

- Em F16, o llama.cpp usa 69–86% da banda: a conta de bolso acerta a ordem de grandeza.
- **Quanto menos bits e mais banda, mais longe do teto.** Em Q4_0 no M2 Ultra sobra só 45%: com 4 bits e
  800 GB/s, desempacotar os pesos e as outras partes do token (atenção, sincronização) passam a pesar.
  Cada bit cortado rende menos velocidade do que a conta promete, e rende menos ainda em máquina rápida.
- Na máquina lenta (M1, 68 GB/s) o Q4_0 fica em 79% do teto. É ali, na máquina popular, que cortar bytes
  vira velocidade quase 1 para 1.

Isso é coerente com o que o STRATA já viu no kernel dos níveis: o tipo D perdeu velocidade por custo de
desempacotar, não por bytes ([`docs/11`](11-kernel-do-tipo-d.md)).

---

## 3. A máquina popular de 2026, em números

**[fato]** Pesquisa de hardware do Steam, setembro de 2026
([Valve](https://store.steampowered.com/hwsurvey/), lida em 31/10/2026). Somas por faixa são **[conta]**.

| VRAM | % dos PCs | RAM do sistema | % dos PCs |
|---|---:|---|---:|
| até 8 GB | 45,9 (8 GB sozinho: 26,7) | até 12 GB | 9,8 |
| 10–12 GB | 15,8 | 16 GB | 37,8 |
| 16 GB | 27,2 | 24 GB | 2,6 |
| 20–22 GB | 2,4 | 32 GB | 42,2 |
| 24 GB ou mais | 7,4 | 48 GB ou mais | 6,6 |

Ressalvas da própria fonte e da imprensa: a Valve corrigiu um erro de leitura de VRAM em fevereiro de 2026 e
parte da virada para 16 GB veio de leituras melhores de placas Radeon
([TweakTown](https://www.tweaktown.com/news/112992/mor), [xenospectrum](https://xenospectrum.com/en/steam-hardware-survey-july-2026/)).
O Steam mede gamers, não "todo PC"; a máquina de escritório média deve ter menos VRAM.

**Bandas típicas (especificação; [conta] onde indicado):**

| Elo | Banda | Fonte |
|---|---:|---|
| RTX 5070, 12 GB GDDR7 | 672 GB/s | [TechPowerUp](https://techpowerup.com/review/nvidia-geforce-rtx-5070-founders-edition) |
| RTX 4070 | 504 GB/s (452 GB/s no README do NWC) | idem / [NWC](https://github.com/parda21/NWC) |
| DDR5-5600, dois canais | 89,6 GB/s | [conta] 5600 MT/s × 8 B × 2 |
| DDR4-3200, dois canais | 51,2 GB/s | [conta] |
| PCIe 4.0 ×16 (GPU ↔ RAM) | ~32 GB/s por sentido, nominal | especificação PCIe |
| SSD NVMe PCIe 4.0 / 5.0 | ~7 / ~14 GB/s em leitura sequencial | especificação de mercado |

**O que cabe onde [conta].** Tamanho só dos pesos, sem KV cache; Q4_K_M ≈ 4,85 bits/peso, Q6_K ≈ 6,56.

| Modelo denso | Q4_K_M | Q6_K | Q8_0 | BF16 |
|---|---:|---:|---:|---:|
| 8B | 4,9 GB | 6,6 GB | 8,5 GB | 16,1 GB |
| 14B | 9,0 GB | 12,1 GB | 15,7 GB | 29,6 GB |
| 32B | 19,9 GB | 26,9 GB | 34,9 GB | 65,6 GB |
| 70B | 42,8 GB | 57,9 GB | 75,0 GB | 141,2 GB |

- Com 8 GB de VRAM (o grupo mais comum), cabe um 8B em Q4/Q6; um 14B já não cabe nem em Q4.
- Com 16 GB de VRAM + 32 GB de RAM (as duas faixas mais comuns; o Steam não publica a combinação), um 70B em Q4_K_M (42,8 GB)
  roda dividido: ~14 GB na GPU e ~29 GB na RAM. **[conta]** O pedaço da RAM limita tudo: 89,6 ÷ 29 ≈
  **3,1 tokens/s de teto**, ~2 tokens/s com 60 GB/s efetivos.
- **MoE muda o gargalo.** O gpt-oss-120b tem 117 B de pesos e 5,1 B ativos por token, em MXFP4, e cabe
  numa GPU de 80 GB; o gpt-oss-20b (21 B, 3,6 B ativos) roda em 16 GB
  ([cartão do modelo](https://huggingface.co/openai/gpt-oss-120b)). **[conta]** Os pesos ativos do 120b
  somam ~2,7 GB por token: na DDR5 de dois canais, teto de ~33 tokens/s, rápido o bastante. O problema
  vira **capacidade**: ~63 GB precisam caber, e só 6,6% dos PCs do Steam têm 48 GB de RAM ou mais.

---

## 4. KV cache: a outra metade dos bytes

**Analogia.** Os pesos são o livro de receitas; o KV cache é o caderno de anotações da conversa. O livro tem
tamanho fixo; o caderno cresce a cada token lido ou escrito.

- **[fato]** Sistemas de serviço anteriores desperdiçavam 60–80% da memória do KV cache por fragmentação e
  reserva antecipada; o PagedAttention do vLLM baixa o desperdício para menos de 4% e dá até 24× mais vazão
  que o HF Transformers ([blog do vLLM](https://vllm.ai/blog/2023-06-20-vllm)) e 2–4× contra
  FasterTransformer e Orca com a mesma latência ([Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180)).
- **[fato]** KIVI (KV cache em 2 bits): 2,6× menos memória de pico no Llama-2-7B, lote até 4× maior e
  2,35–3,47× mais vazão, numa A100 ([Liu et al., ICML 2024](https://arxiv.org/abs/2402.02750)).
- **[fato]** KVQuant: menos de 0,1 de perplexidade a mais com KV em 3 bits; LLaMA-7B com 1 milhão de tokens
  de contexto numa A100-80GB e 10 milhões em 8 GPUs
  ([Hooper et al., NeurIPS 2024](https://arxiv.org/abs/2401.18079)).
- **[fato]** Atenção com uma só cabeça de K/V (multi-query) permite contexto até 32× maior na mesma
  memória ([Pope et al., MLSys 2023](https://arxiv.org/abs/2211.05102)).
- **[fato]** QuantSpec usa um KV cache **hierárquico** INT4 dentro de INT8 (o mesmo cache serve o rascunho
  e a conferência) e chega a 2,5× de velocidade com aceitação acima de 90%
  ([Tiwari et al., ICML 2025](https://arxiv.org/abs/2502.10424)). É a ideia de níveis aninhados do STRATA,
  aplicada ao KV cache, e já está publicada.

**[conta]** Tamanho do KV em FP16 = 2 × camadas × cabeças KV × dimensão da cabeça × 2 bytes por token:

| Modelo | Por token | 32 mil tokens | 128 mil tokens |
|---|---:|---:|---:|
| Llama-3.1-8B (32 camadas, 8 cabeças KV, 128) | 128 KiB | 4 GiB | 16 GiB |
| Llama-3.1-70B (80, 8, 128) | 320 KiB | 10 GiB | 40 GiB |

Com 32 mil tokens, o KV do 8B (4 GiB) já é quase do tamanho dos pesos em Q4_K_M (4,9 GB). Em contexto
longo, um formato que só comprime pesos ataca metade do problema.

---

## 5. Lote, especulativa e por que a casa é diferente do datacenter

- **[fato]** O lote converte espera de memória em trabalho útil: num servidor com uma A100, ir para lote 64
  aumenta a vazão ~14× e a latência ~4× (Databricks, link acima). O kernel Marlin (FP16 × INT4) mantém
  quase os 4× teóricos de ganho até lotes de 16–32 e chega a 2,8× ponta a ponta no vLLM
  ([Frantar et al., PPoPP 2025](https://arxiv.org/abs/2408.11743)).
- **[fato]** Separar pré-preenchimento e decodificação em máquinas diferentes: 7,4× mais pedidos ou SLO
  12,6× mais apertado ([DistServe](https://arxiv.org/abs/2401.09670)); 1,4× mais vazão a 20% menos custo,
  ou 2,35× mais vazão com o mesmo custo e potência ([Splitwise](https://arxiv.org/abs/2311.18677)).
- **[fato]** Decodificação especulativa (um modelo pequeno chuta, o grande confere vários tokens numa
  passada só, com a mesma saída): 2–3× no T5-XXL ([Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192));
  EAGLE-2 relata 3,05–4,26× ([arXiv 2406.16858](https://arxiv.org/abs/2406.16858)).
- **[fato] Especulativa com offload.** Quando o modelo não cabe na VRAM, conferir 1 ou 1000 tokens custa o
  mesmo tempo, porque o tempo é a cópia dos pesos pela PCIe. O SpecExec explora isso: modelos de 50 B+ em
  GPU de consumo com offload para a RAM a 4–6 tokens/s em 4 bits ou 2–3 tokens/s em 16 bits, 10–18× mais
  rápido que a geração sequencial no mesmo hardware; um Llama-2-70B em 16 bits com offload numa RTX 3090
  leva pelo menos 4,5 s por token sem especulação ([Svirschevski et al., NeurIPS 2024](https://arxiv.org/abs/2406.02532)).
- **[fato]** SubSpec usa como rascunho o **próprio modelo**, com as camadas que foram para a RAM trocadas por
  cópias quantizadas em poucos bits na GPU: 9,1× no Qwen2.5-7B com 8 GB de VRAM e 12,5× em média no
  Qwen2.5-32B com 24 GB ([arXiv 2509.18344](https://arxiv.org/abs/2509.18344)). O resumo não diz se a
  conferência transmite as camadas em precisão cheia; pela descrição, sim.

**[conta] Por que o STRATA cortou a especulativa entre níveis e por que isso não vale no offload.** Em
[`docs/08`](08-precisao-sob-demanda.md) (E1) o rascunho era o nível 0 e a conferência o nível 2, os dois na
RAM: aceitação α = 0,886 e custo relativo do rascunho c ≈ 0,71, teto de 1,08×. Com a fórmula padrão
(tokens por rodada = (1 − α^(γ+1)) ÷ (1 − α), custo = γ·c + 1), o melhor ganho é:

| c (custo do rascunho ÷ custo da conferência) | melhor γ | ganho |
|---:|---:|---:|
| 0,71 (os dois na RAM, E1) | 2 | 1,10× |
| 0,22 | 6 | 2,16× |
| 0,10 | 9 | 3,24× |

O c cai quando a conferência é cara (transmitida pela PCIe) e o rascunho é barato (residente na VRAM). É
exatamente o caso offload. A seção 8 (lacuna L1) transforma isso numa proposta.

---

## 6. Compressão sem perdas: quando vira velocidade

**Analogia.** Mandar a mudança num caminhão com as caixas bem arrumadas: cabe mais por viagem, mas alguém
tem de desempacotar na chegada. Só compensa se desempacotar for mais rápido que a viagem extra.

### 6.1 O que está medido

| Trabalho | Formato de entrada | Tamanho | Velocidade em decodificação | Hardware |
|---|---|---|---|---|
| [DFloat11](https://arxiv.org/abs/2504.11651) (NeurIPS 2025) | BF16 | −30%, bit a bit idêntico | ~2× **mais lento** em lote 1 ([README](https://dfloat11-readme.static.hf.space/README.md)); 2,3–46,2× mais vazão contra offload para a CPU; 5,7–14,9× mais contexto | GPUs de datacenter |
| [NWC](https://github.com/parda21/NWC) (README, 2026-09) | BF16 / FP8 | −31% / −13% | Qwen3-4B BF16: 45 → 55 tokens/s (**1,22×**) numa RTX 4070; 1,08× numa A16; FP8 ~igual ao nativo | consumo; não revisado por pares |
| [Doubleword](https://blog.doubleword.ai/faster-than-speed-of-light) (blog, 2026-05) | FP8, só o expoente (tANS) | ~7 bits por peso FP8 | **1,09×** além da banda de pico num kernel de soma; teto ~1,15× para FP8 | RTX 4090 |
| [Huff-LLM](https://arxiv.org/abs/2502.00922) | BF16/FP16 | até −32% | até 31% menos latência e 26% menos energia; com 256 GB/s **não ganha tempo, mas ainda poupa 24% de energia** | simulação + FPGA |
| [EntroLLM](https://arxiv.org/abs/2505.02380) | uint8/uint4 + Huffman | até −30% / −65% | 31,9–146,6% mais rápido | Jetson (borda) |
| ["Approaching Shannon Bound"](https://arxiv.org/abs/2606.15789) (ISCA 2026) | bf16 a int4 | a 0,01–0,1 bit do limite | lote máximo 47 → 75 (Qwen-14B), vazão até 1,2× (Qwen-14B) e 1,6× (Mixtral) | A100 / H200 |
| [ZipNN](https://arxiv.org/abs/2411.05239) | BF16 / FP32 | −33%, às vezes mais de −50% / −17% | só armazenamento e rede | CPU |

### 6.2 O cruzamento com o que o STRATA mediu

- **[fato, STRATA]** A entropia de ordem 0 dos índices de 4 bits nas matrizes Q4_K é **3,86 bits**
  ([`docs/05`](05-resultados.md), seção do Q4_K_M): um codificador de entropia tiraria ~0,12 bit/peso de um
  Q4_K_M, ~2,5% dos bytes.
- **[fato]** O artigo do limite de Shannon diz que formatos de grupo tipo AWQ ficam ~1,1–1,3× acima do limite
  de entropia (seção VI da [versão HTML](https://arxiv.org/html/2606.15789)); a entropia que eles medem é de
  ordem 0 por tensor, como a do STRATA.
- **[conta]** No formato que a máquina popular roda (4 bits com escala escolhida para minimizar erro), o
  ganho de velocidade possível de uma compressão sem perdas fica em ~3%. Os ganhos de 1,1–1,6× acima
  existem porque a entrada é BF16 ou FP8, com 13–30% de redundância. **Sem perdas acelera quem roda BF16 ou
  FP8; quase nada para quem roda Q4.**
- **[hipótese]** A diferença entre 1,04× (Q4_K, STRATA) e 1,1–1,3× (AWQ, artigo) pode vir do quantizador:
  os k-quants escolhem escala e mínimo por sub-bloco para usar a faixa inteira, o AWQ não. Não conferi.
- **[fato]** Energia é diferente de velocidade: o Huff-LLM poupou 24% de energia mesmo quando a banda
  sobrava, porque ler da DRAM domina o gasto. Para bateria (notebook, celular), cortar bytes vale mesmo
  sem ganho de tokens/s.

---

## 7. Preço por token e energia: a escala do problema

- **[fato]** O preço para atingir um mesmo desempenho caiu entre 9× e 900× por ano conforme o teste, mediana
  ~50×/ano; com dados desde 2024, mediana ~200×/ano
  ([Epoch AI](https://epoch.ai/data-insights/llm-inference-price-trends)). Outra análise, com dados da
  Epoch e da Artificial Analysis, acha 5–10×/ano para modelos de fronteira
  ([arXiv 2511.23455](https://arxiv.org/abs/2511.23455)).
- **[fato]** O Google mediu 0,24 Wh por prompt de texto mediano no Gemini, contando aceleradores, CPU e DRAM
  do host, máquinas ociosas e o prédio; só os chips ativos dão 0,10 Wh; queda de 33× em um ano
  ([cobertura da DCD](https://www.datacenterdynamics.com/en/news/google-median-gemini-prompt-uses-024-watt-hours-of-power-and-consumes-026ml-of-water/);
  artigo da empresa, não revisado por pares).
- **[fato]** Em 46 modelos e 1.858 configurações em H100 e B200, o tipo de tarefa muda a energia em até 25×
  (raciocínio contra conversa) e a utilização da GPU em 3–5×
  ([Chung et al., "Where Do the Joules Go?", 2026](https://arxiv.org/abs/2601.22076)). O tamanho da
  resposta pesa mais que a precisão dos pesos.
- **[fato]** Limitar a potência de A100 a 175 W aumentou o tempo de inferência em 6,7% e cortou 23,2% da
  energia ([Samsi et al., 2023](https://arxiv.org/abs/2310.03003); número de um resumo secundário).
- **[fato]** Inferência local em placas Blackwell de consumo (RTX 5060 Ti, 5070 Ti, 5090): US$ 0,001–0,04 por
  milhão de tokens **só de eletricidade**, 40–200× abaixo das APIs baratas; NVFP4 dá 1,6× a vazão do BF16
  com 2–4% de perda de qualidade e 41% menos energia
  ([arXiv 2601.09527](https://arxiv.org/abs/2601.09527)). A conta não inclui o preço da placa.
- **[fato, fonte fraca]** Um desenvolvedor estimou US$ 1,50 por milhão de tokens num MacBook Pro M5 contando
  depreciação e luz, ~3× o preço de um equivalente no OpenRouter
  ([AI Weekly](https://aiweekly.co/alerts/apple-silicon-local-inference-costs-3x-more-than-openrouter)).
- **[fato]** A memória ficou cara em 2026: a TrendForce estimou alta de 90–95% nos contratos de DRAM no 1º
  trimestre e de 58–63% no 2º; um módulo DDR5 de 32 GB passou de ~US$ 95 (set/2025) para ~US$ 400
  (dez/2025) no varejo; a fatia de wafers de DRAM indo para HBM subiu de ~8% (2024) para ~23% (2026)
  (via [Tom's Hardware](https://tomshardware.com/pc-components/dram/the-ram-pricing-crisis-has-only-just-started-team-group-gm-warns-says-problem-will-get-worse-in-2026-as-dram-and-nand-prices-double-in-one-month),
  [TweakTown](https://www.tweaktown.com/news/110829/dram-prices-to-increase-by-up-to-63-percent-in-q2-nand-flash-by-up-to-75-percent-new-report-says/index.html),
  [Fusion Worldwide](https://info.fusionww.com/blog/why-ddr5-prices-have-nearly-quadrupled-in-2026);
  todas secundárias). NAND também subiu 70–75% no 2º trimestre.

**O que isso diz para a visão do STRATA [hipótese].** Concorrer com a nuvem em **preço por token** é uma
corrida que o local perde: o preço da nuvem cai uma ou duas ordens de grandeza por ano e o local paga a
placa e a memória, que subiram em 2026. O que o local oferece de verdade é **viabilidade** (rodar o modelo
grande sem nuvem), privacidade e custo fixo. Em 2026, o byte de RAM é o insumo mais caro da máquina
popular; poupar capacidade de RAM vale mais do que poupar eletricidade.

---

## 8. Lacunas: o que ninguém mediu ou resolveu

Cada lacuna diz o que existe, o que falta, por que importa para o STRATA, quanto custaria testar e o que
mataria a ideia. Em ordem de valor para o STRATA, na minha avaliação.

### L1. Especulativa com o rascunho residente e a conferência transmitindo só os estratos que faltam

- **Existe:** SpecExec e SubSpec mostram 9–18× em offload, conferindo com o modelo completo transmitido
  pela PCIe (seção 5). QuantSpec aninha INT4 em INT8, mas para o KV cache e com tudo na GPU.
- **Falta [hipótese]:** no formato 0.3 do STRATA, o Q4_K_M é o estrato 0 e o BF16 exato é Q4_K_M + níveis +
  fecho (~11,1 bits/peso no total, [`docs/05`](05-resultados.md)). Se o estrato 0 fica na VRAM como
  rascunho, a conferência só precisa receber os ~6,3 bits/peso que faltam, em vez dos 16 bits do BF16:
  **~2,5× menos bytes pela PCIe por conferência** [conta]. Não achei trabalho que transmita só o resíduo de
  precisão para conferir.
- **Por que importa:** é a única rota encontrada que entrega **saída idêntica ao BF16 numa placa de 8–24 GB**
  com velocidade utilizável, que é o item (b) e (d) da visão. E reabre a especulativa entre níveis que o E1
  cortou: com c ≈ 0,1–0,2 em vez de 0,71, o teto sobe de 1,08× para 2–3× [conta, seção 5].
- **Exemplo [conta], 8B numa placa de 8 GB com 272 GB/s e PCIe a 25 GB/s efetivos, tudo no teto ideal e
  supondo α = 0,886 contra o BF16 (o E1 mediu esse α contra o nível 2, não contra o BF16):**

  | Estratégia | Bytes por conferência | Tokens/s com saída idêntica ao BF16 |
  |---|---:|---:|
  | BF16 transmitido inteiro, sem especulativa | 16,1 GB | 1,6 |
  | BF16 transmitido inteiro + rascunho Q4 (tipo SubSpec) | 16,1 GB | 8,2 (γ = 15) |
  | **Só os estratos 1–4 (~6,3 bits/peso) + rascunho Q4 = estrato 0** | 6,3 GB | **14,9** (γ = 10) |
  | Referência: só o Q4_K_M na VRAM (saída não é a do BF16) | – | 55,5 |

- **Ressalva honesta:** comparado a rodar só o Q4_K_M na VRAM, isto é **mais lento** (~15 contra ~55
  tokens/s no exemplo); o ganho é qualidade de BF16, não velocidade. E exige
  um kernel de GPU que reconstrua BF16 a partir de Q4_K + planos + fecho em velocidade de PCIe; o fecho é
  rANS, e decodificar rANS em GPU nessa taxa é o risco técnico.
- **Primeiro teste barato:** só conta, sem kernel. No 8B, medir α com rascunho = Q4_K_M e conferência = BF16
  (o E1 mediu contra o nível 2), e medir quanto tempo a PCIe leva para levar os estratos 1–4 de uma camada.
  Se α (contra BF16) < 0,8 ou se o fecho não decodifica a ≥ 25 GB/s de entrada, a ideia morre.

### L2. Teste direto STRATA contra a deduplicação do Hugging Face (Xet) no mesmo conjunto

- **Existe:** Xet guardou as 29 quantizações do gemma-2-9b-it-GGUF (191 GB) em ~97 GB num ambiente de
  teste ([Hugging Face, "From Chunks to Blocks"](https://huggingface.co/blog/from-chunks-to-blocks)); o
  Hub tem mais de 3,5 PB de GGUF. O STRATA mediu −66% no conjunto BF16 + Q4_K_M + Q5_K_M + Q6_K de 8B e 70B.
- **Falta:** ninguém comparou os dois no **mesmo** conjunto de arquivos. Os 50% do Xet vêm de um repositório
  que inclui F32 e F16 e muitas quantizações parecidas; os 66% do STRATA vêm de outro conjunto.
- **Por que importa:** é a pergunta que um engenheiro do Hub faria primeiro. Se o Xet já pega boa parte do
  ganho, a vantagem do STRATA no armazenamento é a diferença, não os 66%.
- **Custo:** baixo; dá para rodar o chunker de código aberto do Xet localmente no 8B (o container cabe).
- **Mata a ideia se:** o Xet chegar perto de −66% no mesmo conjunto.

### L3. Custo real em lote 1 na máquina popular, com divisão GPU + RAM

- **Existe:** tabelas de comunidade (llama.cpp #4167 para Apple), artigos em A100/H100/4090, e o estudo de
  placas Blackwell de consumo (seção 7).
- **Falta:** medida sistemática de tokens/s, MBU e joules por token na combinação mais comum do Steam
  (8–16 GB de VRAM + 16–32 GB de DDR4/DDR5) com modelos que **não cabem** na VRAM (14B–70B, MoE de 30–120 B).
  O teto da conta (seção 3) diz ~2–3 tokens/s para um 70B; não achei medida publicada que confirme com
  joules. A thread de motores locais (`docs/14`) pode cobrir os tokens/s.
- **Por que importa:** é o cenário da visão (b). Sem ele, qualquer promessa de "70B em PC popular" é conta.

### L4. Compressão sem perdas no trecho lento (RAM → GPU ou SSD → RAM), não na VRAM

- **Existe:** NWC, DFloat11, Huff-LLM e o artigo do limite de Shannon decodificam **dentro da GPU**, onde a
  banda é de centenas de GB/s e o decodificador mal acompanha (DFloat11 fica 2× mais lento em lote 1).
- **Falta [hipótese]:** no offload, o gargalo é a PCIe (~25–32 GB/s) ou o SSD (7–14 GB/s), 10 a 100× mais
  lentos que a VRAM. Um decodificador que não acompanha a VRAM acompanha a PCIe com folga. Não achei
  medida de compressão sem perdas aplicada só ao trecho transmitido, onde cada 30% a menos de bytes viraria
  quase 30% a mais de velocidade.
- **Limite honesto:** vale para BF16 e FP8; para Q4 o teto é ~3% (seção 6). Então é útil junto com L1
  (transmitir resíduos BF16) ou para quem roda FP8, não para o usuário de Q4 puro.

### L5. Capacidade de RAM como custo, com preços de 2026

- **Existe:** análises de custo local (seção 7) contam eletricidade e preço da placa; algumas são de antes da
  alta da DRAM.
- **Falta:** uma conta de custo total que trate a **RAM** como o recurso caro (subiu ~4× no varejo entre
  set/2025 e dez/2025 segundo fontes secundárias) e compare: comprar mais RAM, paginar do SSD
  ([LLM in a flash](https://arxiv.org/abs/2312.11514): modelos até 2× a DRAM; [PowerInfer-2](https://arxiv.org/abs/2406.06282):
  47 B num celular a 11,68 tokens/s com 19 GB livres e 2,13 com 7 GB) ou guardar menos bytes.
- **Por que importa:** muda o argumento do STRATA de "armazenamento mais barato" para "menos RAM comprada".
- **Ressalva:** LLM in a flash e PowerInfer dependem de esparsidade de ativação (ReLU); os modelos SwiGLU
  densos de hoje são menos esparsos ([PowerInfer](https://arxiv.org/abs/2312.12456) só suporta ReLU/ReGLU).

### L6. Custo por tarefa, não por token

- **Existe:** raciocínio gasta até 25× a energia de uma conversa por resposta (seção 7).
- **Falta:** medir se um modelo maior em 4 bits que acerta com menos tokens de raciocínio sai mais barato
  **por tarefa resolvida** que um menor em 8 bits que pensa mais. A lei de escala de k bits diz que, para
  o mesmo total de bits, 4 bits maximiza a precisão zero-shot
  ([Dettmers e Zettlemoyer, ICML 2023](https://arxiv.org/abs/2212.09720)), mas foi medida antes dos
  modelos de raciocínio e não conta tokens gerados.
- **Por que importa pouco para o STRATA hoje:** é uma pergunta de escolha de modelo, não de formato. Fica
  registrada.

### O que parecia lacuna e não é

- **KV cache com níveis aninhados:** QuantSpec já faz INT4 dentro de INT8 (seção 4).
- **Rascunho quantizado do próprio modelo com offload:** SubSpec já faz (seção 5). O que sobra é L1
  (transmitir só o resíduo).
- **Compressão sem perdas acelerando lote 1:** já medida em BF16/FP8 (seção 6). Para Q4 o STRATA já tem a
  medida que mostra teto de ~3%.

---

## 9. Plano sugerido, em ordem

1. **L2 (Xet contra STRATA), barato e decide o argumento de armazenamento.** Container, 8B, mesmo conjunto
   de arquivos da seção 31 de `docs/05`.
2. **L1, só a conta (sem kernel).** α do Q4_K_M contra o BF16 no 8B e tempo de PCIe dos estratos por
   camada. Escrever a regra de parada antes de rodar, como no `docs/10`.
3. **L3/L4** dependem do que a thread de motores locais (`docs/14`) e a de MoE/offload trouxerem.

O motor de compressão (`docs/10`) continua sendo a prioridade definida pelo Cantyll; L1 e L2 não competem
com ele por GPU, porque o primeiro passo de cada um roda no container.

---

## Fontes

Artigos: [2402.16363](https://arxiv.org/abs/2402.16363) · [2403.14123](https://arxiv.org/abs/2403.14123) ·
[2503.08311](https://arxiv.org/abs/2503.08311) · [2211.05102](https://arxiv.org/abs/2211.05102) ·
[2309.06180](https://arxiv.org/abs/2309.06180) · [2402.02750](https://arxiv.org/abs/2402.02750) ·
[2401.18079](https://arxiv.org/abs/2401.18079) · [2502.10424](https://arxiv.org/abs/2502.10424) ·
[2408.11743](https://arxiv.org/abs/2408.11743) · [2401.09670](https://arxiv.org/abs/2401.09670) ·
[2311.18677](https://arxiv.org/abs/2311.18677) · [2211.17192](https://arxiv.org/abs/2211.17192) ·
[2406.16858](https://arxiv.org/abs/2406.16858) · [2406.02532](https://arxiv.org/abs/2406.02532) ·
[2509.18344](https://arxiv.org/abs/2509.18344) · [2504.11651](https://arxiv.org/abs/2504.11651) ·
[2502.00922](https://arxiv.org/abs/2502.00922) · [2505.02380](https://arxiv.org/abs/2505.02380) ·
[2606.15789](https://arxiv.org/abs/2606.15789) · [2411.05239](https://arxiv.org/abs/2411.05239) ·
[2511.23455](https://arxiv.org/abs/2511.23455) · [2601.22076](https://arxiv.org/abs/2601.22076) ·
[2310.03003](https://arxiv.org/abs/2310.03003) · [2601.09527](https://arxiv.org/abs/2601.09527) ·
[2312.11514](https://arxiv.org/abs/2312.11514) · [2406.06282](https://arxiv.org/abs/2406.06282) ·
[2312.12456](https://arxiv.org/abs/2312.12456) · [2212.09720](https://arxiv.org/abs/2212.09720).

Repositórios e páginas: [llama.cpp #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) ·
[NWC](https://github.com/parda21/NWC) · [DFloat11](https://dfloat11-readme.static.hf.space/README.md) ·
[Doubleword](https://blog.doubleword.ai/faster-than-speed-of-light) ·
[vLLM](https://vllm.ai/blog/2023-06-20-vllm) · [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) ·
[Epoch AI](https://epoch.ai/data-insights/llm-inference-price-trends) ·
[Steam](https://store.steampowered.com/hwsurvey/) · [Hugging Face Xet](https://huggingface.co/blog/from-chunks-to-blocks) ·
[gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b).

**Não encontrado ou não conferido:** os números de joules por token do "From Words to Watts" (só achei o
resumo); a tabela original do Horowitz (usei a citação da New Electronics); o número de 14 tokens/s do
KTransformers no DeepSeek-V3 (anúncio no Reddit, sem medida independente; deixado para a thread de MoE).
