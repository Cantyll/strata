# Motores de inferência local: o que existe, quanto rende e onde ninguém olhou

Levantamento feito em 2026-10-31, a pedido do Cantyll: o que os projetos do GitHub e os artigos já fizeram para baixar o custo de rodar modelos em máquina comum, com números reais, e onde estão as combinações que ninguém testou. Só entra número com link. Quando o número é conta minha e não medida, está escrito "conta".

Este documento cobre os **motores** (os programas que rodam o modelo). Duas outras frentes da mesma pesquisa ficam em documentos próprios: MoE, offload e paginação de pesos em detalhe, e quantização abaixo de 4 bits. Aqui elas aparecem só onde cruzam com um motor ou com o STRATA.

## 1. A regra que explica quase tudo

Na geração (um token por vez, uma pessoa usando), o computador lê **todos os pesos ativos uma vez por token**. A conta é de padaria: se a estrada da memória entrega 100 GB/s e o modelo tem 5 GB, o teto é 20 tokens/s, não importa quantos núcleos você tem.

Os números do llama.cpp confirmam. Llama 2 7B em Q4_0 (3,56 GiB = 3,82 GB), geração de 128 tokens, tudo na GPU:

| máquina | banda (GB/s) | medido (tok/s) | teto pela banda (conta) | quanto do teto |
|---|---:|---:|---:|---:|
| Apple M1 (8 núcleos GPU) | 68 | 14,15 | 18 | 80% |
| Apple M4 (10) | 120 | 24,11 | 31 | 77% |
| Apple M4 Pro (20) | 273 | 50,74 | 71 | 71% |
| Apple M4 Max (40) | 546 | 83,06 | 143 | 58% |
| Apple M2 Ultra (76) | 800 | 94,27 | 209 | 45% |
| RTX 4060 Ti | 288 | 64,03 | 75 | 85% |
| RTX 3060 12 GB | 360 | 76,92 | 94 | 82% |
| RTX 3090 | 936 | 161,89 | 245 | 66% |
| RTX 4090 | 1008 | 188,96 | 264 | 72% |
| RTX 5090 | 1792 | 300,40 | 469 | 64% |

Fontes: [Apple Silicon, discussão #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) e [CUDA, discussão #15013](https://github.com/ggml-org/llama.cpp/discussions/15013). Bandas das placas NVIDIA são as de fábrica.

**O que isso diz:**

- Nas máquinas populares (M1, M4, RTX 3060, 4060 Ti), o llama.cpp já chega a **77–85% do teto**. Não há 2× escondido no kernel: o único jeito de ir bem mais rápido é **ler menos bytes por token**.
- Nas máquinas caras (M2 Ultra, 5090), sobra 35–55%: ali o gargalo já é cálculo e sincronização, não a memória.
- Por isso o STRATA mede tudo em **bits por peso**: cada bit a menos é velocidade e RAM, na mesma proporção.

**Quando o modelo não cabe**, a estrada muda e fica muito mais estreita:

| de onde vêm os pesos que faltam | banda típica | 1 GB por token custa |
|---|---:|---:|
| RAM → GPU pela PCIe 4.0 x16 | ~25–32 GB/s (teórico 32) | ~0,03–0,04 s |
| SSD NVMe → RAM | ~3 GB/s (980 PRO medido no [HOBBIT](https://arxiv.org/html/2411.01433)) a 17,5 GB/s (SSD do M3 Max, [flash-moe](https://github.com/danveloper/flash-moe)) | ~0,06–0,33 s |

Analogia: a RAM é a geladeira, o SSD é o mercado do bairro. Cozinhar um prato por token buscando ingrediente no mercado é dez a cem vezes mais lento. Todos os projetos abaixo são, no fundo, jeitos de ir menos ao mercado.

## 2. Os projetos, um por um

### Motores gerais (o modelo cabe, ou quase)

| projeto | o que faz | número real (hardware) | limite | relação com o STRATA |
|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | o padrão: GGUF, k-quants, CPU/GPU, `-ngl` divide camadas entre GPU e CPU, `--n-cpu-moe` manda especialistas para a CPU, `mmap` lê do disco sob demanda | tabela da seção 1; gpt-oss-120b (~64 GB) numa RTX 5090 com 21 camadas MoE na CPU: ~30 tok/s ([guia #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)) | `mmap` de modelo maior que a RAM vira leitura aleatória do SSD (0,87 tok/s no teste do PR #25294, abaixo) | o STRATA já tem os níveis como tipos do ggml (P 43–46, D 47–50, [docs/11](11-kernel-do-tipo-d.md)) e o estrato 0 é o Q4_K_M do Hub byte a byte |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | fork do autor dos k-quants: tipos IQ_K, IQ_KS, treliça (IQ1_KT a IQ4_KT), MLA para DeepSeek, `--override-tensor` desde fev/2025, Hadamard no cache KV | o README não traz número com hardware; remete à wiki | só CPU AVX2/NEON e CUDA Turing+ | os tipos de treliça são a fronteira de qualidade por bit que os níveis teriam de alcançar; fora do escopo daqui (ver o documento de quantização) |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | formato EXL3, "variante enxuta do QTIP" (treliça), Hessiana calculada na conversão, cache KV de 2 a 8 bits, paralelismo entre GPUs de consumo | os números são imagens no README; não copio | só CUDA na prática; conversão leva horas para 70B+ | o EXL3 também usa a Hessiana, como o C1 do STRATA ([docs/10](10-motor-de-compressao.md)) |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | compila o modelo (TVM) para Vulkan, Metal, CUDA, ROCm, WebGPU, iOS e Android | sem número no README | um formato próprio por alvo | é o caminho para celular e navegador, onde o STRATA não tem kernel (ARM sem NEON) |
| [llamafile](https://github.com/Mozilla-Ocho/llamafile) | llama.cpp + Cosmopolitan num executável só | sem número no README atual | Windows só roda executável < 4 GB | distribuição, não custo |
| [vLLM `--cpu-offload-gb`](https://docs.vllm.ai/en/latest/configuration/engine_args.html) | trata a RAM como extensão da VRAM | a doc avisa: "precisa de interconexão CPU-GPU rápida", e a parte na RAM é copiada **a cada passada** | pensado para servidor com lote | nenhuma |

### CPU e bits muito baixos

| projeto | o que faz | número real | limite |
|---|---|---|---|
| [bitnet.cpp](https://github.com/microsoft/BitNet) | kernels para modelos de 1,58 bit (ternários) | 2,37–6,17× mais rápido em x86, 1,37–5,07× em ARM; "100B numa CPU a 5–7 tok/s" | o modelo precisa nascer ternário; o maior oficial tem 2,4B; o de 100B não está na lista de modelos |
| [T-MAC](https://github.com/microsoft/T-MAC) | multiplica pesos de 1–4 bits por tabela (sem desquantizar), na CPU | Llama-2-7B 4 bits no Snapdragon X Elite: 12,6 tok/s em 2 núcleos, 18,7 em 4 (a NPU dá 10,4); Jetson Orin, 2 bits: 15,62 tok/s a 0,66 J/token contra 7,08 tok/s e 2,12 J do llama.cpp na CPU | ganha mais quanto menos bits; EuroSys 2025 |

### O modelo não cabe: dividir entre GPU, CPU e disco

| projeto | ideia | número real (hardware) | limite |
|---|---|---|---|
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | poucos neurônios "quentes" disparam quase sempre: ficam na GPU; os frios rodam na CPU | média 13,20 tok/s, pico 29,08, numa RTX 4090; até 11,69× o llama.cpp (Falcon-40B FP16) e até 3× no Llama 2 70B | **só modelos com ReLU** (ReluLLaMA, ProSparse, Bamboo); Llama, Mistral e Qwen originais não servem |
| [PowerInfer-2](https://arxiv.org/abs/2406.06282) | o mesmo no celular, com NPU + CPU e cache de neurônios | TurboSparse-Mixtral-47B a 11,68 tok/s num celular | precisa do modelo re-treinado esparso (TurboSparse, ~90% de esparsidade, ~US$ 0,1 mi) |
| [LLM in a flash](https://arxiv.org/abs/2312.11514) (Apple) | pesos no flash; lê só as linhas dos neurônios previstos ativos, em blocos contíguos | modelos de até 2× a RAM; 4–5× mais rápido que a leitura ingênua na CPU, 20–25× na GPU | sem código oficial; depende de esparsidade prevista |
| [TEAL](https://arxiv.org/abs/2408.14690) | esparsidade de ativação sem treino (zera as menores) | 40–50% de esparsidade, 1,53× e 1,8× na geração, Llama 2/3 e Mistral 7B–70B | "compatível com quantização de pesos" segundo o resumo |
| [AirLLM](https://github.com/lyogavin/airllm) | uma camada por vez na GPU, o resto no disco | 70B em ~4 GB de VRAM; **não publica tok/s** | conta: 70B em FP16 são ~140 GB lidos por token; a 3 GB/s de NVMe, ~47 s por token. Pede ~360 GB de disco para o modelo fatiado do Flash-Next |
| [FlexGen](https://arxiv.org/abs/2303.06865) | GPU + CPU + disco, pesos e cache KV em 4 bits, agenda para **vazão** | OPT-175B numa GPU de 16 GB a 1 token/s **com lote de 144** | não serve para uma pessoa conversando: é vazão de lote |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | especialistas MoE na CPU (AMX/AVX-512, NUMA), atenção e partes densas na GPU | DeepSeek-R1/V3 671B, Q4_K_M, 1× RTX 4090D + 2× Xeon Gold 6454S, 382 GB a 1 TB de RAM: 13,69 tok/s na geração, 97 no prefill; llama.cpp no mesmo: 4,51 e 10,31 ([tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)) | "máquina comum" com 382 GB de RAM e Xeon de servidor |
| [mixtral-offloading](https://github.com/dvmazur/mixtral-offloading) ([artigo](https://arxiv.org/html/2312.17238)) | cache LRU de especialistas na GPU, pré-carga, HQQ (atenção 4 bits, especialistas 2–3 bits) | Mixtral-8x7B, especialistas 2 bits: 2,09 tok/s no T4 do Colab, 2,28 na RTX 3060; o `accelerate` ingênuo dá 0,66 e 0,92 | qualidade dos especialistas a 2–3 bits |
| [Fiddler](https://arxiv.org/abs/2402.07033) | em vez de trazer o especialista para a GPU, leva a ativação para a CPU e calcula lá | Mixtral-8x7B sem compressão, GPU de 24 GB: "mais de 3 tok/s"; 8,2–10,1× os sistemas de offload | — |
| [HOBBIT](https://arxiv.org/abs/2411.01433) | no erro de cache, carrega uma **versão de precisão menor** do especialista menos importante; feito sobre o llama.cpp | Jetson Orin: 13,0× o llama.cpp no Mixtral; RTX 4090 + CPU: 1,31× o llama.cpp | guarda as precisões como **versões separadas** (FP16 + INT4 na 4090, INT8 + INT2 no Orin); o artigo não mede o custo extra |
| [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) | especialistas lidos do SSD com cache próprio e E/S assíncrona (O_DIRECT), saída igual à sem streaming | GLM-5.2 Q2_K_XL (254 GB) numa GB10 de 128 GB: 2,20 tok/s com 79 GB de cache (79% de acerto) contra 0,87 tok/s do `mmap` + `--cpu-moe` | aberto, não aceito; precisão fixa |
| [flash-moe](https://github.com/danveloper/flash-moe) | C + Metal, lê só os 4 especialistas ativos do SSD, cache = page cache do sistema (~71% de acerto) | Qwen3.5-397B-A17B num MacBook M3 Max de 48 GB: 4,36 tok/s em 4 bits (209 GB); 5,74 tok/s em 2 bits (120 GB), **mas o 2 bits quebra o JSON das chamadas de ferramenta** | precisão fixa por arquivo |
| [Guanaco / qwen3.8flashnext-ssdstreaming](https://github.com/neutron42-jp/qwen3.8flashnext-ssdstreaming) | `madvise` + io_uring + pré-busca da camada seguinte | Qwen3.8-Flash-Next (103 GB, IQ4_XS) numa RTX 5060 8 GB, limite de 24 GB de RAM: 8,8 tok/s (12,0 com top-k mudado) | o 12,0 muda o modelo (top-k 8) |
| ["The Other Half of the Memory Wall"](https://arxiv.org/abs/2609.18063) | prevê o roteamento da próxima camada um token antes e usa a previsão como roteamento | MoE de 35B (19,5 GB em 4 bits) a ~20 tok/s numa máquina de 24 GB, 3 GiB de memória ativa | re-treina (cabeça de previsão + LoRA de recuperação) |
| [SpecExec](https://arxiv.org/abs/2406.02532) | decodificação especulativa com árvores grandes, feita para offload | modelos 50B+ em GPU de consumo com offload: 4–6 tok/s em 4 bits, 2–3 em 16 bits; até 20 tokens por passada do modelo grande | rascunho é um modelo separado |

### Várias máquinas

| projeto | ideia | número real | limite |
|---|---|---|---|
| [distributed-llama](https://github.com/b4rtaz/distributed-llama) | paralelismo de tensor por Ethernet; RAM dividida entre nós | sem tok/s no README (remete às discussões) | nós em potência de 2; Q40 com sincronização Q80 |
| [exo](https://github.com/exo-explore/exo) | junta os aparelhos da casa; escolhe a divisão pela banda entre eles | tensor paralelo: 1,8× com 2 aparelhos, 3,2× com 4; RDMA por Thunderbolt 5 | foco em Mac; Linux só CPU |
| [Petals](https://github.com/bigscience-workshop/petals) | estilo BitTorrent: cada voluntário guarda umas camadas | Llama 2 70B até 6 tok/s, Falcon 180B até 4 | depende do enxame estar vivo; o README não mostra data recente |

## 3. O que os números juntos dizem

1. **Quando o modelo cabe, o software já está perto do teto** nas máquinas populares (77–85%). Ganho grande só vem de menos bytes por peso, que é a frente do STRATA ([docs/10](10-motor-de-compressao.md)), e lá os níveis ainda ficam de −0,03 a +0,14 bit/peso da fronteira dos k-quants.
2. **Quando não cabe, a vitória vem de ler menos do disco**, não de calcular mais rápido: neurônios quentes (PowerInfer), linhas ativas (LLM in a flash), especialistas em cache (mixtral-offloading, flash-moe, PR #25294), previsão do que vem (Guanaco, "Other Half"), ou conferir vários tokens por leitura (SpecExec).
3. **Os MoE grandes mudaram o jogo do SSD.** Com só 4 de 512 especialistas ativos, um 397B roda a 4,4 tok/s num notebook de 48 GB. Num modelo denso, a mesma conta não fecha.
4. **Ninguém trata a precisão como algo que se lê aos pedaços.** Todos os projetos de offload usam precisão fixa por arquivo. Os dois que trocam de precisão em tempo de execução (HOBBIT, PagedWeight) ou guardam cópias separadas, ou ficam só entre GPU e RAM. Esse é o espaço do STRATA.
5. **Três "milagres" têm letra miúda:** AirLLM (cabe, mas não publica velocidade; pela conta, dezenas de segundos por token num 70B), FlexGen (1 tok/s só com lote de 144) e bitnet.cpp (100B a 5–7 tok/s, mas não existe modelo ternário de 100B para baixar).

## 4. Lacunas: combinações que nenhum projeto testou

Critério: aparece em pelo menos dois trabalhos separados, mas ninguém juntou; e o STRATA já tem uma peça. Para cada uma: o que já existe, por que pode valer, a conta, o risco e o primeiro teste barato. "Não encontrei" quer dizer que procurei e não achei, não que não existe.

### L1. Especulativa entre níveis quando o modelo não cabe (o resíduo vem do disco)

**O que já existe.** O STRATA mediu a especulativa entre níveis com tudo na RAM e cortou: α = 0,886 no 1B, teto 1,08× ([docs/08, seção 2.4 e 5](08-precisao-sob-demanda.md)). O SpecExec mostra que, com offload, conferir muitos tokens por leitura dá 4–6 tok/s em modelos de 50B+. O SPEQ ([arXiv 2510.18525](https://arxiv.org/abs/2510.18525)) e o QuantSpec ([arXiv 2502.10424](https://arxiv.org/abs/2502.10424)) já usam "rascunho = bits altos, conferência = todos os bits", mas num acelerador próprio (SPEQ) ou só no cache KV (QuantSpec).

**O que ninguém juntou.** Base (Q4_K_M do Hub) na RAM ou VRAM como rascunho; planos de resíduo no SSD ou na RAM do outro lado da PCIe; uma leitura dos planos confere γ tokens. Não encontrei isso em motor nenhum.

**Por que muda a conta.** O que matou a ideia em 2026-10-26 foi c, o custo do rascunho dividido pelo custo da conferência: com tudo na RAM, c = 0,71. Se o resíduo vem de uma estrada 10 a 30 vezes mais estreita, c cai muito. Pela mesma fórmula do docs/08, com o α medido de 0,886 (conta, não medida):

| c | melhor γ | ganho sobre rodar o nível de cima lendo o resíduo a cada token |
|---:|---:|---:|
| 0,71 (tudo na RAM, medido) | 2 | ~1,1× |
| 0,30 | 5 | 1,8× |
| 0,10 | 9 | 3,2× |
| 0,05 | 12 | 4,4× |

A saída tem a distribuição exata do nível de cima: qualidade de Q6 com a RAM de um Q4.

**Risco honesto.** (a) O concorrente não é "ler o resíduo a cada token", é o `-ngl` parcial do llama.cpp, que já manda o que não cabe para a CPU. Conta para um 32B numa GPU de 24 GB: o Q6_K tem ~27 GB, sobram ~3 GB na CPU, e o `-ngl` parcial deve dar uns 13 tok/s; a especulativa com o resíduo pela PCIe dá na mesma faixa. **Na fronteira GPU/RAM, provavelmente empata.** A janela real é RAM/SSD, onde o `mmap` desaba (0,87 tok/s no PR #25294). (b) α no 8B não foi medido. (c) Precisa do layout separado (base + planos) também na memória, com kernel que soma o plano à base; os tipos P de hoje guardam o código inteiro e não servem de rascunho.

**Primeiro teste (container, 1B, horas).** Simular a estrada estreita com um limite de leitura (cgroup de I/O ou `mmap` com o resíduo num arquivo à parte) e medir tok/s de: nível 2 lendo o resíduo a cada token, especulativa entre níveis com γ = 4–12, e `-ngl`/`mmap` comum do Q6_K. Régua: ganha se ≥ 1,5× o melhor dos dois concorrentes com a mesma RAM.

### L2. Especialistas MoE em um arquivo só, com precisão lida sob demanda

**O que já existe.** flash-moe e o PR #25294 leem especialistas do SSD em precisão fixa. O flash-moe mostra o dilema: 2 bits é 32% mais rápido (5,74 contra 4,36 tok/s) e cabe mais no cache, mas quebra o JSON; 4 bits é correto e mais lento. O HOBBIT troca de precisão no erro de cache, mas com **cópias separadas**. O D2MoE ([arXiv 2504.15299](https://arxiv.org/abs/2504.15299)) aninha os bits dos especialistas (matryoshka) no aparelho, até 1,39× de vazão e −53% de memória de pico, sem SSD. O PagedWeight ([arXiv 2607.16184](https://arxiv.org/abs/2607.16184)) guarda os especialistas em planos de bits e os move entre GPU e RAM no vLLM (servidor com lote, sem SSD).

**O que ninguém juntou.** Um arquivo no SSD com base de 2 ou 4 bits e planos de resíduo; o cache do sistema guarda a base de muitos especialistas; os planos só são lidos para os especialistas de peso alto no roteador (ou para os quentes). Disco: 209 GB em vez de 120 + 209 = 329 GB se as duas precisões forem guardadas, no exemplo do flash-moe (conta). Não encontrei em motor do GitHub.

**Por que o STRATA tem a peça.** O formato 0.3 já é exatamente base do Hub + planos de 1 bit + fecho exato, com SHA-256 por plano ([README](../README.md)). O leitor em fluxo já lê planos separados.

**Risco honesto.** Os níveis do STRATA custam de −0,03 a +0,14 bit/peso contra k-quants independentes no 8B denso. MoE nunca foi medido no STRATA, e a memória do projeto anota que a compressão ainda não rendeu em MoE: começar só se o Cantyll pedir. O ganho depende de quanto os especialistas de peso baixo toleram 2 bits; o próprio flash-moe mostra que o modelo inteiro em 2 bits não tolera.

**Primeiro teste.** Num MoE pequeno (Qwen3-30B-A3B ou OLMoE), medir a KL de: tudo 4 bits, tudo 2 bits, e "top-1 do roteador em 4 bits, resto em 2" usando os níveis como base + 1 plano. Se a mistura ficar perto de tudo-4-bits, vale montar o leitor.

### L3. Esparsidade de ativação × planos de resíduo

**O que já existe.** O LLM in a flash lê do flash só as linhas dos neurônios previstos ativos; o TEAL zera 40–50% das ativações sem treino e ganha 1,53–1,8×; o PowerInfer separa neurônios quentes e frios, mas só em modelos ReLU.

**O que ninguém juntou.** Base completa na RAM; plano de resíduo lido só para as linhas ativas (ou só para as quentes). Os neurônios frios ficam no Q4, os ativos sobem para Q6. Não encontrei.

**Por que pode valer.** Se 50% das linhas estão desligadas no token, o resíduo custa metade dos bytes. Combina com L1 e L2.

**Risco honesto.** Em SwiGLU (Llama, Qwen) a esparsidade é aproximada, não exata como em ReLU; o TEAL mede a esparsidade nas entradas das matrizes, e o ganho real em kernel de CPU com blocos de 256 pesos pode sumir na granularidade.

**Primeiro teste.** No 1B, medir a KL de "nível 2 só nas linhas onde a ativação passa do limiar do TEAL, nível 0 no resto" contra o nível 0 e o nível 2 inteiros. Barato, só PyTorch, sem kernel.

### L4. Precisão por camada escolhida a cada token, dentro do GGUF

**O que já existe.** O DP-LLM ([NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115920)) escolhe a precisão de cada camada a cada passo sobre o Any-Precision LLM; o PagedWeight faz o mesmo por lote. Ambos fora do ecossistema GGUF/llama.cpp.

**O que falta.** Um motor popular que faça isso com o Q4_K_M do Hub como base. É mais engenharia que pesquisa, e só rende se L1 ou L3 mostrarem que ler menos planos compensa. Fica em último.

## 5. O que não é lacuna (já está feito ou não fecha)

- **Kernel mais rápido para o modelo que cabe.** O llama.cpp já chega a 77–85% da banda nas máquinas populares.
- **Camada por camada do disco para modelo denso** (AirLLM). Pela conta, dezenas de segundos por token; não fica usável.
- **Offload para vazão de lote** (FlexGen, vLLM). Resolve outro problema (servidor), não o da pessoa com um notebook.
- **Juntar máquinas** (exo, distributed-llama, Petals). Já existe e funciona; o STRATA não tem peça nova aqui.
- **Modelo ternário** (bitnet.cpp). Exige treinar do zero; não converte modelo existente, o que bate de frente com a visão (c) do projeto.

## 6. Ordem sugerida

1. **L1 no container, 1B** (horas, sem Colab): é a única lacuna que reaproveita uma medida já feita (α = 0,886) e mira direto a visão (b) e (d) do projeto, rodar além da RAM.
2. **L3 no container, 1B** (PyTorch, horas): se der certo, multiplica L1 e L2.
3. **L2** só se o Cantyll quiser abrir MoE; o primeiro teste cabe num MoE pequeno.
4. **L4** depois de 1 e 2.

Nada disso compete com o motor de compressão (docs/10), que segue sendo a prioridade: L1 e L2 usam os níveis como estão, mas ficam melhores a cada décimo de bit que o motor ganhar.

## Fontes

Motores: [llama.cpp](https://github.com/ggml-org/llama.cpp) · [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) · [ExLlamaV3](https://github.com/turboderp-org/exllamav3) · [MLC LLM](https://github.com/mlc-ai/mlc-llm) · [llamafile](https://github.com/Mozilla-Ocho/llamafile) · [vLLM, opções do motor](https://docs.vllm.ai/en/latest/configuration/engine_args.html) · [bitnet.cpp](https://github.com/microsoft/BitNet) · [T-MAC](https://github.com/microsoft/T-MAC) · [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) · [AirLLM](https://github.com/lyogavin/airllm) · [KTransformers](https://github.com/kvcache-ai/ktransformers) e [tutorial DeepSeek](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md) · [mixtral-offloading](https://github.com/dvmazur/mixtral-offloading) · [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) · [flash-moe](https://github.com/danveloper/flash-moe) · [Guanaco](https://github.com/neutron42-jp/qwen3.8flashnext-ssdstreaming) · [distributed-llama](https://github.com/b4rtaz/distributed-llama) · [exo](https://github.com/exo-explore/exo) · [Petals](https://github.com/bigscience-workshop/petals)

Medidas do llama.cpp: [Apple Silicon #4167](https://github.com/ggml-org/llama.cpp/discussions/4167) · [CUDA #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) · [gpt-oss #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)

Artigos: [FlexGen](https://arxiv.org/abs/2303.06865) · [LLM in a flash](https://arxiv.org/abs/2312.11514) · [PowerInfer-2](https://arxiv.org/abs/2406.06282) · [Mixtral offloading](https://arxiv.org/abs/2312.17238) · [Fiddler](https://arxiv.org/abs/2402.07033) · [HOBBIT](https://arxiv.org/abs/2411.01433) · [TEAL](https://arxiv.org/abs/2408.14690) · [SpecExec](https://arxiv.org/abs/2406.02532) · [SPEQ](https://arxiv.org/abs/2510.18525) · [QuantSpec](https://arxiv.org/abs/2502.10424) · [D2MoE](https://arxiv.org/abs/2504.15299) · [PagedWeight](https://arxiv.org/abs/2607.16184) · [The Other Half of the Memory Wall](https://arxiv.org/abs/2609.18063) · [DP-LLM](https://neurips.cc/virtual/2025/poster/115920)
