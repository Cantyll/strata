# 15. MoE, offload e paginação de pesos: o que já foi medido e onde há lacunas

Pesquisa feita em 2026-10-31, a pedido do Cantyll. Pergunta: como se roda hoje um modelo maior que a RAM ou a VRAM (paginação de especialistas de MoE, offload para SSD, previsão de especialistas, esparsidade de ativação, cache, leitura parcial), quantos tokens por segundo isso dá em cada hardware, e onde um formato em estratos com leitura parcial, como o STRATA, ajudaria de verdade.

Convenção do texto: **[fonte]** é número tirado de artigo, repositório ou relato, com link na seção 9. **[conta]** é conta minha a partir das configurações públicas dos modelos. **[inferência]** é opinião minha, sem medida. Relatos de fórum vão marcados como **[relato]**: não têm revisão nem condições controladas.

Fica de fora: KV cache, batching, roofline geral, custo por token e energia (estão no docs/13), quantização abaixo de 4 bits (docs/12) e comparação de motores locais (docs/14).

## 1. Resumo

- **Na geração, um token custa ler os pesos ativos.** A velocidade é, em primeira ordem, a banda do meio onde os pesos estão dividida pelos bytes lidos por token. Num MoE só os especialistas escolhidos são lidos, e isso é o que torna o offload viável [conta, seção 2].
- **Hoje, em PC comum, o que funciona é deixar os especialistas na RAM e calculá-los na CPU** (Fiddler, KTransformers, `--n-cpu-moe` do llama.cpp). O DeepSeek-V3 671B em Q4 chega a 13,69 tok/s, mas numa máquina com dois Xeon e 1 TB de DDR5 [fonte]. A barreira continua sendo a **capacidade** da RAM: o DeepSeek-V3 em Q4 pede uns 380 GB.
- **Abaixo da RAM, no SSD, os números despencam.** Ficam entre 0,4 e 2 tok/s para modelos de 100 a 670 B, quase tudo em relatos e provas de conceito de 2025 e 2026 [relato]. O artigo mais recente desse regime em desktop (FlashMoE, 2026) nem publica tok/s absolutos [fonte].
- **O jeito de ler pesa tanto quanto a banda.** O mmap paginando em 4 KB tirou 377 MB/s de um NVMe que lê 2,98 GB/s em sequência, 8 vezes menos [relato]. No celular, a UFS vai de 450 MB/s em blocos de 4 KB a 4 GB/s em blocos de 512 KB [fonte].
- **Trocar o especialista que faltou por uma versão de menos bits já foi testado e funciona.** O HOBBIT perde no máximo 1 ponto de acurácia e lê até 4 vezes menos [fonte]. Mas ele guarda **duas cópias separadas** de cada especialista.
- **Especialistas aninhados em planos de bits já existem desde 15/10/2026 (DPA-MoE),** mas só com tudo residente na GPU. O artigo exclui de propósito o regime de offload e de SSD [fonte].
- **A lacuna mais clara é o que nenhum desses junta** [inferência]: um cache cuja unidade é o par (especialista, plano de bits), não o especialista inteiro. Cada falta leria só os planos de cima, e cada especialista quente subiria de precisão lendo só o plano que falta. O STRATA já tem quase tudo isso (planos por tensor, troca a quente bit a bit). Falta o que importa para MoE: tensores 3-D, o kernel `MUL_MAT_ID` e níveis que não percam para os k-quants.

## 2. A conta básica: bytes por token

Na geração com lote 1, cada token lê uma vez cada peso ativo. Se os pesos estão num meio de banda B (GB/s) e o token precisa de V GB, o teto é B / V tokens por segundo. O cálculo e a atenção somem nessa escala; o que manda é mover bytes.

Tamanho dos especialistas roteados, tirado do `config.json` de cada modelo, com cada especialista = 3 matrizes (gate, up, down) e uns 4,5 bits por peso (Q4_K) [conta]:

| modelo | especialistas por camada (usados) | camadas MoE | pesos por especialista | 1 especialista em Q4 | todos os especialistas em Q4 | especialistas lidos por token em Q4 |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 128 (8) | 48 | 4,7 M | 2,7 MB | 16,3 GB | 1,02 GB |
| gpt-oss-120b | 128 (4) | 36 | 24,9 M | 14 MB | 64,5 GB | 2,02 GB |
| Mixtral-8x7B | 8 (2) | 32 | 176 M | 99 MB | 25,4 GB | 6,34 GB |
| Qwen3-235B-A22B | 128 (8) | 94 | 18,9 M | 10,6 MB | 128 GB | 7,98 GB |
| DeepSeek-V3/R1 | 256 (8) | 58 | 44 M | 24,8 MB | 368 GB | 11,5 GB |

O gpt-oss-120b vem publicado em MXFP4 (≈ 4,25 bits), então a linha dele já é quase o tamanho real.

Teto de tok/s **só pela leitura dos especialistas**, sem cache e com 80% de acerto no cache [conta]:

| meio (banda usada na conta) | Qwen3-30B-A3B | gpt-oss-120b | Qwen3-235B | DeepSeek-V3 |
|---|---|---|---|---|
| DDR5 dois canais (80 GB/s) | 78 | 40 | 10 | 7,0 |
| PCIe 4.0 x16 (25 GB/s efetivos) | 25 / 123 | 12 / 62 | 3,1 / 16 | 2,2 / 11 |
| NVMe Gen4, leitura sequencial (7 GB/s) | 6,9 / 34 | 3,5 / 17 | 0,9 / 4,4 | 0,6 / 3,0 |
| NVMe paginado em 4 KB (0,38 GB/s, medido abaixo) | 0,4 / 1,9 | 0,2 / 0,9 | 0,05 / 0,24 | 0,03 / 0,16 |

Na linha da DDR5 não há cache: os pesos já estão lá. Nas outras, o primeiro número é sem cache e o segundo com 80% de acerto (só 20% dos especialistas vêm do meio lento).

Três leituras dessa tabela [inferência]:

1. **No SSD, só a taxa de acerto do cache salva.** Sem cache, nenhum modelo grande passa de 1 tok/s. Com 80% de acerto, os do meio da tabela chegam a 3-17 tok/s.
2. **O tamanho do especialista decide o padrão de leitura.** O Qwen3-30B tem especialistas de 2,7 MB (≈ 0,9 MB por matriz), e o Mixtral de 99 MB. Os dois são grandes o bastante para leitura sequencial. Paginar em 4 KB joga fora 95% da banda (última linha).
3. **Cada bit por peso a menos na falta é velocidade quase linear,** desde que a falta domine o tempo. É aí que entra a precisão por especialista (seção 5).

**Conferência com o que foi medido:**
- KTransformers, DeepSeek-V3 em Q4, um soquete com 220 GB/s medidos, 8 especialistas: 8,73 tok/s medidos [fonte] contra um teto de 220 / 11,5 ≈ 19 tok/s [conta]. Isso dá 46% da banda.
- Com dois soquetes e 6 especialistas, foram 13,69 tok/s [fonte], contra um teto de 220 / 8,6 ≈ 25 tok/s por soquete (55%) [conta].
- A conta acerta a ordem de grandeza, e as implementações boas chegam a cerca de metade do teto.

## 3. A hierarquia de memória, com números medidos

| meio | banda | onde foi medido |
|---|---|---|
| DDR5, por soquete Xeon 8452Y | 220 GB/s (125 GB/s cruzando soquetes) | KTransformers, SOSP 2025 [fonte] |
| PCIe 3.0 x16 / 4.0 x16 | 32 / 64 GB/s nominais | Fiddler [fonte] |
| SSD do M1 Max, leitura densa / esparsa / esparsa em pacotes | 6,10 / 1,25 / 2,25 GB/s | LLM in a Flash [fonte] |
| NVMe PCIe 3.0, sequencial / leitura sobreposta na geração / mmap paginado | 2,98 / 2,80 / 0,377 GB/s | llama.cpp, discussão #23324 [relato] |
| UFS 4.0 (OnePlus 12), blocos de 4 KB / 512 KB | 0,45 / 4 GB/s | PowerInfer-2 [fonte] |
| NVMe PCIe 5.0 (SK hynix P51) | 7,4 GB/s de leitura | FlashMoE [fonte] |

O LLM in a Flash recomenda pedaços de 32 KiB ou mais, lidos por várias threads [fonte]. Os números de PowerInfer-2 e do mmap dizem o mesmo: **o tamanho de cada leitura importa tanto quanto o disco.**

## 4. O que foi medido, por família de técnica

### 4.1 Especialistas na RAM, calculados na CPU

A ideia é não mover o peso: mover a ativação (alguns KB) é muito mais barato que mover um especialista de centenas de MB pelo PCIe. O Fiddler mede que transferir o peso leva de 2 a 5 vezes o tempo de calcular, e que copiar a ativação fica abaixo de 1% da latência [fonte].

| sistema | modelo e precisão | hardware | geração (tok/s) | fonte |
|---|---|---|---|---|
| Fiddler | Mixtral-8x7B, 16 bits (> 90 GB) | uma GPU de 24 GB + Xeon | > 3; 1,26× o llama.cpp em média | [fonte] |
| KTransformers V0.2 | DeepSeek-V3/R1 Q4_K_M, 6 especialistas | 2× Xeon Gold 6454S, 1 TB DRAM, 4090D 24 GB | 13,69 (llama.cpp: 4,51) | [fonte] |
| KTransformers V0.2 | o mesmo, 8 especialistas, um soquete | 382 GB DRAM, ≥ 14 GB VRAM | 8,73 | [fonte] |
| KTransformers (SOSP 2025) | DS-V3, DS-V2.5, Qwen2-57B; BF16 na A100, Int4 na 4080 | 2× Xeon 8452Y, 1 TB DDR5 | 1,25 a 1,76× o llama.cpp em BF16; com o "Expert Deferral", 1,66 a 2,56× | [fonte] |
| llama.cpp `--n-cpu-moe 24` | gpt-oss-120b | RTX 3090 24 GB + 64 GB DDR4-3200 | ≈ 15 | [relato] |

O "Expert Deferral" do KTransformers adia alguns especialistas para a camada seguinte, para a CPU e a GPU trabalharem ao mesmo tempo. Ganha até 45% e muda as notas em no máximo 2 pontos [fonte]. **Muda o modelo**, então não é exato.

### 4.2 Offload com cache e previsão de especialistas

Aqui o especialista é copiado (da RAM para a VRAM, ou do SSD para a RAM) quando faz falta, e duas coisas cortam o custo. A primeira é um cache dos especialistas recentes. A segunda é prever os da camada seguinte e trazê-los antes.

| sistema | ideia | modelo e hardware | resultado | fonte |
|---|---|---|---|---|
| Mixtral-offloading (Eliseev e Mazur, 2023) | cache LRU por camada (k = 2 em 12 GB, k = 4 em 16 GB) + carga especulativa com o roteador da camada seguinte aplicado ao estado atual; especialistas em HQQ de 2 ou 3 bits | Mixtral-8x7B: T4 do Colab, RTX 3060, 3080 Mobile, A100 | 2 bits: 2,09 (T4), 2,28 (3060), 2,66 (3080M) tok/s. Sem cache e sem previsão: 1,17 / 1,55 / 1,76. Offload ingênuo do `accelerate`: 0,66 / 0,92 / 1,06 | [fonte] |
| Pre-gated MoE (ISCA 2024) | o roteador do bloco N já escolhe os especialistas do bloco N+1 (exige retreino) | Switch Transformer | só 23% mais lento que ter tudo na GPU | [fonte] |
| MoE-Infinity (2024-2025) | rastreia que especialistas cada pedido usa para prever e guardar | DeepSeek e Mixtral em máquina pessoal | 3,1 a 16,7× menos latência por token que vLLM, Ollama, DeepSpeed e BrainStorm | [fonte] |
| ProMoE (2024) | cache proativo com pré-busca em pedaços | RTX 4090 24 GB | geração 1,66× o llama.cpp em média | [fonte] |
| AdapMoE (2024) | número de especialistas por token adaptativo + pré-busca | borda | 25% menos especialistas ativos, 1,35× mais rápido | [fonte] |
| EdgeMoE (2023) | bits por especialista + pré-carga por estatística entre camadas | Jetson TX2 e outros | 1,19 a 2,77×, menos de 2% de perda | [fonte] |
| FlashMoE (jan. 2026) | especialistas no SSD, um arquivo por especialista, substituição no cache aprendida por ML | Ryzen 9600X, RTX 5070 Ti 16 GB, 64 GB DDR5, NVMe PCIe 5.0; OLMoE-1B-7B e Qwen3-30B-A3B | acerto até 51% acima de LRU/LFU; até 2,6× outros sistemas; carregar um especialista ≈ 3 ms contra ≈ 158 µs de cálculo; **sem tok/s absolutos** | [fonte] |
| llama.cpp, discussão #23324 (maio 2026) | N vagas por camada em memória compartilhada do Metal, LRU, faltas lidas com `pread`, saída exata | Qwen3-30B-A3B Q6_K, M3 Pro 36 GB | 38,1 tok/s com tudo residente. 64 vagas (16 GB): 27,5. 16 vagas (7,8 GB): 15,7. 8 vagas (6,3 GB): 12. Num M1 Pro de 16 GB: 13 | [relato] |
| o mesmo fio, outro usuário | pool de leitura assíncrona + especialistas fixos na VRAM | GLM-4.5-Air Q4_K_M (73,5 GB) num PC de 32 GB, CUDA | mmap: 0,43 tok/s. Pool: 0,75. Fixar 24 GB numa máquina de 32 GB: caiu para 0,12 | [relato] |
| DeepSeek-R1 671B por mmap do NVMe | só o KV cache na RAM; o cache de páginas do sistema guarda os pesos usados | quant. dinâmica, 96 GB de RAM, NVMe Gen5 x4, sem GPU | ≈ 2,1 tok/s com 2 K de contexto; gargalo no SSD, CPU a 30% | [relato] |

O que as previsões acertam: a carga especulativa do Mixtral-offloading olha 1 camada à frente. Dois números chegaram só de segunda mão, por um trecho de busca atribuído a um artigo posterior que não consegui conferir: o AdapMoE passaria de 80% de acerto na previsão, e com a VRAM apertada o acerto do cache do ProMoE cairia para menos de 50%. O FlashMoE cita, na motivação, 86% para o ótimo de Belady e 73% para o LRU, num cenário que não especifica [fonte]. No fio do llama.cpp, fixar os especialistas calibrados em uns 10 tokens deu 42% de acerto a frio, e 56% junto com o LRU [relato].

### 4.3 Precisão por especialista na hora da falta

É a família mais próxima do STRATA.

| sistema | como guarda as precisões | resultado | fonte |
|---|---|---|---|
| **HOBBIT** (nov. 2024) | **cópias separadas**: float16 + int4 na RTX 4090, int8 + int2 no Jetson AGX Orin; todos os especialistas na RAM da CPU; caches separados para alta e baixa precisão; o especialista menos importante que faltar vem na versão de menos bits | integrado ao llama.cpp. Geração no Orin: 13× o llama.cpp e 3,64× o MoE-Infinity (Mixtral-8x7B). Na 4090: 3,21× o MoE-Offloading e 0,99× o Fiddler (Mixtral). Acurácia cai no máximo 1% (GSM8K do Mixtral 0,52 → 0,51). Não mede o custo de guardar as duas cópias | [fonte] |
| EdgeMoE (2023) | bits escolhidos por especialista, fixos | ver 4.2 | [fonte] |
| DynaExq (nov. 2025) | especialistas quentes em alta precisão, o resto em baixa; promoção assíncrona pelo histórico do roteador | Qwen3-80B: 73,09% → 77,57% contra PTQ estática, com a mesma memória; até 2,73× em lote 32 | [fonte] |
| DyMoE (mar. 2026) | quantização dinâmica por importância + camadas sensíveis preservadas + pré-busca | primeiro token 3,44 a 22,7× mais rápido, geração até 14,58×, memória de 12 a 24 GB | [fonte] |
| **DPA-MoE** (15/10/2026) | **planos de bits aninhados** (AnyBCQ com escala compartilhada, grupos de 128, de 2 a 8 planos). Subir de k para k+1 planos lê só o plano novo e a escala dele | até 4,5× menos KL que k uniforme com o mesmo tamanho; 4 a 8% mais lento na geração. **Tudo residente na GPU (96 GB). SSD, cache no host e offload ficam explicitamente fora** | [fonte] |

Custo do aninhamento no DPA-MoE: um especialista do Qwen3-30B-A3B que serve de 2 a 8 bits ocupa 5.184 KB. São 4.608 KB de planos e 576 KB de escalas [fonte]. O AnyBCQ padrão ocupa 37,5% a mais.

### 4.4 Esparsidade de ativação em modelos densos

Aqui não há especialistas: o modelo prevê, a cada token, que linhas ("neurônios") da FFN vão sair diferentes de zero, e lê só essas.

| sistema | exige | hardware e modelo | resultado | fonte |
|---|---|---|---|---|
| Deja Vu (ICML 2023) | previsor por camada | OPT-175B | mais de 2× o FasterTransformer | [fonte] |
| LLM in a Flash (Apple, ACL 2024) | **ativação ReLU** (OPT 97%, Falcon re-treinado com ReLU 95%, Llama 2 com FATReLU 90%) | M1 Max, metade do modelo na DRAM | OPT-6.7B: 3.182 → 669 ms/token na CPU e 2.218 → 84 ms na GPU | [fonte] |
| PowerInfer (SOSP 2024) | neurônios quentes na GPU, frios na CPU | RTX 4090; OPT até 175B | até 11,69× o llama.cpp | [fonte] |
| PowerInfer-2 (2024) | modelos esparsificados (TurboSparse) para o ganho grande | OnePlus 12 (24 GB, UFS 4.0) | TurboSparse-Mixtral-47B: 11,68 tok/s com 19 GB livres e 2,13 tok/s com 7 GB. Mistral-7B SiLU com metade da FFN no flash: 5,3 tok/s | [fonte] |
| Ripple (2024) | reorganiza no flash os neurônios que ativam juntos | 3 celulares | até 4,32× de banda efetiva e 5,93× menos latência de E/S; com metade dos pesos no flash, 71,9 a 97,7% da latência era E/S | [fonte] |
| TEAL (2024) | nada: sem treino | Llama-2, Llama-3, Mistral, 7 a 70 B | 40 a 50% de esparsidade, 1,53 a 1,8× mais rápido | [fonte] |

**Limite desta família** [inferência a partir das fontes]: os ganhos de 10× ou mais vêm de modelos com ReLU ou re-treinados para esparsidade. Os modelos que as pessoas baixam hoje (Llama 3, Qwen 3, Mistral) usam SwiGLU e, sem treino, chegam a 40-50% (TEAL). Isso corta metade dos bytes, não 90%.

### 4.5 Ler o modelo inteiro de fora, comprimido

- **FlexGen (ICML 2023):** OPT-175B numa GPU de 16 GB, juntando GPU, CPU e disco, com pesos e KV em 4 bits. Chega a 1 token/s de vazão, mas com lote efetivo de 144. Serve para processamento em lote, não para conversa [fonte].
- **DFloat11 (2025):** pesos BF16 sem perdas em ≈ 70% do tamanho, descomprimidos na GPU. Comparado a mandar parte do modelo sem compressão para a CPU, gera de 2,3 a 46,2 vezes mais tokens por segundo [fonte]. É a prova de que **compressão sem perdas também é velocidade quando o gargalo é mover bytes**. O formato 0.2 do STRATA (10,59 bits/peso) está na mesma faixa, mas decodifica na CPU.

### 4.6 Comprimir os especialistas uns contra os outros

- **D²-MoE (ICML 2025):** cada especialista vira uma base comum da camada (média ponderada por Fisher) + um delta de posto baixo (SVD). Supera outros compressores em mais de 13% com 40-60% de compressão, sem treino [fonte]. **Tem perdas.** Não achei quem use isso para offload, com a base na RAM e só os deltas vindo do SSD.

## 5. Onde o STRATA ajudaria de verdade, e onde não

O que o STRATA tem hoje e importa aqui (do próprio repo):
- planos de bits gravados por tensor, em ordem estrato-maior; o leitor sabe ler um prefixo, e o D2 guarda um arquivo por estrato (docs/02, docs/05 seção 19);
- troca a quente do nível de um tensor dentro do llama.cpp, bit a bit idêntica, em ≤ 16 µs, e subida cooperativa entre tokens (docs/08, E3 e E4);
- SHA-256 por stream, conferido antes de usar (A.13).

O que não tem:
- o modo residual 0.3 só aceita tensores BF16 **2-D**, e no GGUF os especialistas são tensores 3-D (`ffn_*_exps`);
- os tipos D e livro não têm `MUL_MAT_ID`, o kernel de MoE (kernel/README.md);
- os níveis ainda perdem para os k-quants com a mesma RAM: +0,11 a +0,23 bit/peso no 8B, e o aninhamento custa ≈ 0,18 b por nível no livro de códigos (memória do projeto, docs/09 a 11).

Nos arquivos do Hugging Face, Qwen3, Mixtral e DeepSeek guardam cada especialista como tensores 2-D próprios. Então, no STRATA, cada plano de cada especialista já é um stream separado, que pode ser lido com `pread` ou HTTP Range. Tamanho de um plano de 1 bit por matriz [conta]: 197 KB no Qwen3-30B-A3B, 1,8 MB no DeepSeek-V3, 7 MB no Mixtral. Todos acima dos 32 KiB do LLM in a Flash e perto ou acima dos 512 KB em que a UFS satura. **O layout já serve; o que falta é kernel e tipo de tensor.**

Onde **ajuda**, em ordem de força [inferência]:

1. **Cache por plano, não por especialista.** A memória rápida guarda os planos 0..k de cada especialista, com k alto nos quentes e baixo nos frios. Na falta, lê-se só a base. Quando um especialista esquenta, sobe-se de nível lendo só o plano que falta, e não uma segunda cópia inteira, como no HOBBIT e no DynaExq. É a junção de HOBBIT (falta em baixa precisão funciona, ≤ 1% de perda) com DPA-MoE (planos aninhados funcionam, 4-8% de custo de kernel) no regime que o DPA-MoE deixou de fora.
2. **Fazer a falta ler menos bytes.** Ler 2 bits em vez de 4 corta pela metade os bytes de cada falta. No regime em que o SSD domina (Qwen3-235B ou GLM-4.5-Air num PC de 32 GB), isso é até ~2× a velocidade [conta, seção 2]. O HOBBIT já ganha isso com cópias separadas. A vantagem do aninhamento sobre elas é menor: não guardar as duas versões no cache, e a subida custar só o plano novo.
3. **Um arquivo só para todas as precisões, inclusive a exata.** O modo residual já guarda Q4_K_M + planos + fecho até o BF16 exato em ≈ 11,1 bits/peso. Num MoE em offload, o disco guarda o modelo exato e a RAM escolhe a profundidade por especialista. Hoje isso pede um GGUF por precisão.

Onde **não ajuda**, ou ainda não:

- **Quando tudo cabe na RAM** (DeepSeek-V3 numa máquina de 1 TB, gpt-oss-120b num PC de 64 GB): o gargalo é a banda da RAM, e níveis que perdem para os k-quants **pioram** a qualidade com a mesma memória. O caminho do KTransformers e do `--n-cpu-moe` ganha.
- **Nos modelos densos com esparsidade de ativação,** o ganho grande exige ReLU ou re-treino. Nos modelos SwiGLU normais, a leitura parcial por linha corta só 40-50% dos bytes, e o STRATA não tem nada próprio a oferecer ali.
- **Nos 671 B num PC de 32 GB:** mesmo dobrando a vazão da falta, sai de ~0,5 para ~1 tok/s [conta]. Isso não é uso interativo. A meta realista é a faixa em que o conjunto de especialistas é de 1 a 3 vezes a RAM.

## 6. LACUNAS

Cada lacuna diz o que existe (fato), o que falta (inferência minha), o que o STRATA traria e o primeiro teste. "Não achei" quer dizer não achei nas buscas desta pesquisa (arXiv, GitHub, Hugging Face, fóruns), não que não exista.

**L1. Cache de especialistas por plano de bits, em offload (RAM↔VRAM ou SSD↔RAM).**
- Existe: falta em baixa precisão com cópias separadas (HOBBIT, DynaExq, EdgeMoE) e planos aninhados com tudo residente (DPA-MoE, Any-Precision LLM em modelos densos).
- Não achei: ninguém mede planos aninhados com cache e falta de verdade, nem uma política que decida "quantos planos deste especialista ficam na memória".
- O STRATA traria os planos por especialista já em streams separados e a troca a quente bit a bit (E3).
- Primeiro teste: o de baixo custo na seção 7.
- Risco: o HOBBIT com duas cópias pode empatar, se o ganho de não duplicar for pequeno perto da RAM total.

**L2. Leitura do SSD alinhada ao especialista e ao plano, para modelos normais no PC.**
- Existe: o FlashMoE (desktop, um arquivo `.pt` por especialista, sem tok/s absolutos) e as provas de conceito do llama.cpp (maio de 2026: 0,43 → 0,75 tok/s no GLM-4.5-Air; 377 MB/s pelo mmap contra 2,98 GB/s em sequência).
- Falta: medida revisada, com tok/s absolutos, em PCs comuns de 16 a 64 GB, para modelos de 100 a 250 B sem re-treino.
- O STRATA traria o D2, que já separa por estrato com índice e hash. Um layout com a base de cada especialista contígua e os planos em seguida deixa uma leitura sequencial por falta.

**L3. Base comum da camada na RAM + deltas por especialista vindos do SSD.**
- Existe: o D²-MoE comprime (base + delta SVD, com perdas). A genealogia do STRATA (B.3, Δ estratificado) e o BitDelta fazem o mesmo para fine-tunes.
- Não achei: quem use essa decomposição para offload, em que uma falta lê só o delta. Menos ainda com fecho exato.
- Risco alto: se os especialistas forem diversos (o próprio D²-MoE parte dessa diversidade), o delta não fica pequeno sem perder qualidade. Medir antes a energia do delta por camada é barato (CPU, sem rodar o modelo).

**L4. Precisão por neurônio de acordo com a frequência de ativação, em modelos densos.**
- Existe: neurônios quentes e frios por *lugar* (PowerInfer: GPU ou CPU; LLM in a Flash: DRAM ou flash).
- Não achei: quem varie a *precisão* das linhas pela frequência de ativação (linhas frias com menos planos).
- Lacuna fraca: depende de esparsidade que os modelos SwiGLU não têm sem treino. Prioridade baixa.

**L5. O que não é lacuna.** Previsão de especialistas (Pre-gated, ProMoE, AdapMoE, MoE-Infinity, FlashMoE), especialistas calculados na CPU (Fiddler, KTransformers, llama.cpp) e compressão sem perdas para offload (DFloat11) já estão bem cobertos. Entrar nisso seria repetir trabalho.

## 7. Próximo passo proposto (barato, no contêiner)

Regra escrita antes de medir, como no resto do projeto. É uma proposta: nada foi rodado.

**Experimento M1, simulador de tráfego por plano:**
1. Gravar o roteamento real (que especialistas cada token usa, por camada) do Qwen3-30B-A3B em ~2.000 tokens de wikitext e C4. O GGUF Q4_K_M tem ~18 GB; no contêiner de 15 GB ele roda por mmap, devagar, mas só o roteamento interessa. Como alternativa, o OLMoE-1B-7B.
2. Simular, para orçamentos de RAM de 25%, 40% e 60% dos especialistas, os bytes lidos por token em três políticas:
   - (a) LRU de especialistas inteiros em Q4;
   - (b) HOBBIT: Q4 + uma cópia de 2 bits, com a falta de baixa prioridade em 2 bits;
   - (c) cache por plano: base de 2 bits + planos de 1 bit até 4, com profundidade por calor.
3. Para (b) e (c), estimar a qualidade com KL contra o Q4, pondo nos especialistas a precisão que a política teria em cada token.

**Critério de sucesso:**
- (c) lê pelo menos 25% menos bytes por token que (b), com KL no máximo igual, em dois dos três orçamentos.
- Se não ganhar de (b), L1 morre e fica registrado que cópias separadas bastam.

**Custo:** gravar o roteamento leva horas de CPU, e o simulador é só Python. A KL de (b) e (c) exige especialistas com bits trocados por token: no 30B isso pede o Colab, e o 1B-7B roda no contêiner.

**Custo de oportunidade** [inferência]: a prioridade declarada é o motor de compressão (C2 do 8B, docs/10). L1 só vale a pena se os níveis do STRATA chegarem à fronteira dos k-quants. Senão, o cache por plano entrega velocidade com uma qualidade por byte que um GGUF comum já supera. O M1 é a forma de saber se a pista existe sem tirar tempo do motor: nenhum kernel novo antes de o simulador dizer que vale.

## 8. O que esta pesquisa não verificou

- Os tok/s absolutos do HOBBIT, do Fiddler e do FlashMoE estão só em figuras; usei as razões que o texto dá.
- Os números do KTransformers V0.2 vêm do tutorial do próprio repositório, e a manchete dele (11,26 e 74,36 tok/s) não bate com a tabela da mesma página (13,69 e 97,32).
- Os relatos do Reddit e do fio do llama.cpp não têm condições controladas. Os de gpt-oss-120b chegaram por um agregador, não pelo post original.
- O DPA-MoE saiu há duas semanas e não tem revisão nem código conferido aqui.
- Os acertos de previsão do AdapMoE (> 80%) e do ProMoE (< 50% com pouca VRAM) vieram de segunda mão.

## 9. Fontes

Artigos:
- LLM in a Flash: [arXiv 2312.11514](https://arxiv.org/abs/2312.11514) (ACL 2024)
- PowerInfer: [arXiv 2312.12456](https://arxiv.org/abs/2312.12456) (SOSP 2024)
- PowerInfer-2: [arXiv 2406.06282](https://arxiv.org/abs/2406.06282)
- Ripple: [arXiv 2410.19274](https://arxiv.org/abs/2410.19274)
- Deja Vu: [arXiv 2310.17157](https://arxiv.org/abs/2310.17157) (ICML 2023)
- TEAL: [arXiv 2408.14690](https://arxiv.org/abs/2408.14690)
- Mixtral-offloading (Eliseev e Mazur): [arXiv 2312.17238](https://arxiv.org/abs/2312.17238)
- Fiddler: [arXiv 2402.07033](https://arxiv.org/abs/2402.07033) · [código](https://github.com/efeslab/fiddler)
- KTransformers (SOSP 2025): [PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) · [tutorial DeepSeek-R1/V3](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)
- Pre-gated MoE (ISCA 2024): [arXiv 2308.12066](https://arxiv.org/abs/2308.12066)
- MoE-Infinity: [arXiv 2401.14361](https://arxiv.org/abs/2401.14361)
- ProMoE: [arXiv 2410.22134](https://arxiv.org/abs/2410.22134)
- AdapMoE: [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- EdgeMoE: [arXiv 2308.14352](https://arxiv.org/abs/2308.14352)
- HOBBIT: [arXiv 2411.01433](https://arxiv.org/abs/2411.01433)
- DynaExq: [arXiv 2511.15015](https://arxiv.org/abs/2511.15015)
- DyMoE: [arXiv 2603.19172](https://arxiv.org/abs/2603.19172)
- DPA-MoE: [arXiv 2610.21378](https://arxiv.org/abs/2610.21378)
- FlashMoE: [arXiv 2601.17063](https://arxiv.org/abs/2601.17063)
- FlexGen: [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) (ICML 2023)
- DFloat11: [arXiv 2504.11651](https://arxiv.org/abs/2504.11651)
- D²-MoE: [arXiv 2502.17298](https://arxiv.org/abs/2502.17298) (ICML 2025)

Repositórios e relatos:
- llama.cpp, "RFC+PoC: MoE offload to disk with on-demand paging": [discussão #23324](https://github.com/ggml-org/llama.cpp/discussions/23324)
- r/LocalLLaMA, "DeepSeek R1 671B over 2 tok/sec without GPU": [post](https://www.reddit.com/r/LocalLLaMA/comments/1idseqb/)
- gpt-oss-120b com `--n-cpu-moe` numa RTX 3090: [agregador](https://scuttle.klotz.me/tags/gpt-oss+llama.cpp) (post original não conferido)

Configurações dos modelos (seção 2): `config.json` de [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) e [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b).
