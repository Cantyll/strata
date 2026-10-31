# 16 · Síntese das quatro pesquisas: o que se cruza, o que se contradiz e o que testar primeiro

Síntese feita em 2026-10-31, a pedido do Cantyll, sobre as quatro pesquisas do mesmo dia:

| doc | tema | PR (rascunho) | branch |
|---|---|---|---|
| [12](12-quantizacao-sub4-e-codecs.md) | quantização abaixo de 4 bits e codecs de pesos | #22 | `claude/project-thread-evpvzr` |
| [13](13-custo-de-inferencia.md) | custo de inferência (banda, KV, especulativa, preço) | #20 | `claude/project-thread-dvd3a8` |
| [14](14-motores-de-inferencia-local.md) | motores de inferência local do GitHub | #19 | cópia em `/mnt/project-files/pesquisa/` |
| [15](15-moe-offload-paginacao.md) | MoE, offload e paginação de pesos | #21 | `claude/project-thread-ng6vmr` |

Os links para 12–15 só funcionam depois que os quatro PRs entrarem na `main`.

Nada foi medido de novo aqui. Este documento cruza números que já estavam nos quatro textos e nos
resultados do STRATA (`docs/05`, `docs/08` a `docs/11`). Cada afirmação leva uma etiqueta:

- **[fato]**: número publicado ou medido pelo STRATA, com o documento de onde veio.
- **[conta]**: aritmética minha sobre fatos. Dá para refazer com os números citados.
- **[hipótese]**: inferência sem medida. É o que os experimentos da seção 6 decidem.

---

## 1. Resumo em sete frases

1. **As quatro pesquisas concordam no essencial.** Gerar texto custa ler os pesos ativos a cada token.
   Nas máquinas populares o llama.cpp já chega a 77–85% da banda [fato, 14 §1], então o ganho
   grande vem de ler menos bytes ou de ler de um jeito melhor.
2. **A régua do STRATA está ~1 bit atrás do melhor que existe.** Os níveis empatam com os k-quants
   com imatrix, e o trellis (EXL3) gasta 0,5 a 1,5 bit a menos no mesmo KL [fato + conta, 12 §4].
   Esse melhor só roda em GPU NVIDIA, e o alvo do MVP é CPU com llama.cpp.
3. **A especulativa com resíduo (L1 dos docs 13 e 14) perde força quando se faz a conta inteira.**
   Na fronteira GPU/RAM, ela empata com o `-ngl` parcial e fica com metade da velocidade de só
   rodar o Q4_K_M. Na fronteira RAM/SSD de modelo denso, ela não chega a 1 token/s
   [conta, seção 4.2]. Ela sai do topo da lista.
4. **"Passar da barreira da RAM" é, na prática, um problema de MoE.** Os únicos casos medidos de modelo
   maior que a RAM em velocidade de conversa são MoE com especialistas lidos do SSD [fato, 14 §2 e 15 §4].
   O STRATA ainda não lê tensores 3-D nem tem o kernel `MUL_MAT_ID`, e a frente de MoE está fechada até
   o Cantyll pedir.
5. **No SSD, o jeito de ler pesa mais que os bits.** Mudar só o padrão de leitura deu 2,5×; cortar
   1,74× dos bytes deu 1,32× [fato + conta, seção 3, K2].
6. **Armazenamento sem perda está saturado, e o rival mais forte do STRATA é a receita.** Todos os
   codecs chegam a ~66% do BF16. Guardar BF16 + imatrix e regenerar os GGUFs byte a byte já ganha do
   B.10 para quem hospeda [fato, `docs/05` §32]. O teste contra o Xet precisa ter três colunas.
7. **Ordem recomendada:** (1) Xet × B.10 × receita no mesmo conjunto; (2) trellis contra k-quant na
   CPU, se o Cantyll autorizar compilar o ik_llama.cpp; (3) trellis aninhado numa gaussiana. A
   frente de MoE começa por um pré-teste de horas, mas só se o Cantyll abrir essa frente.

---

## 2. Uma régua só para os níveis do STRATA

Os quatro documentos citam a distância dos níveis à fronteira dos k-quants com números diferentes.
Isso não é erro: cada um cita uma variante diferente dos níveis. Todas são do Llama-3.1-8B [fato].

| variante dos níveis | onde | nível 1 | nível 2 | citada por |
|---|---|---:|---:|---|
| tipos D (PR #10), sem GPTQ | `docs/11`, resultados §kernel | +0,23 b | +0,11 b | doc 15 |
| P e linha com GPTQ (C1, PR #15) | `docs/10` | −0,02 a +0,14 b (wiki / C4) | −0,03 a +0,09 b | doc 14 ("−0,03 a +0,14") |
| livro de códigos aninhado (PR #13) | `docs/09` | +0,006 b | −0,027 b | doc 12 |

**Leitura honesta [conta]:** o melhor que o STRATA mediu é **empate** com os k-quants com imatrix
(dentro de ±0,03 b no livro; até +0,14 b no C4 com GPTQ). O alocador C2 passou na triagem do 1B e
falta o 8B (`docs/10`). Nenhuma variante está à frente da curva dos k-quants.

**Regra para os próximos textos:** sempre dizer qual variante e qual texto (wiki ou C4), e sempre dois
números de bits: só as matrizes de camada (comparável com artigos) e o arquivo inteiro (o que se paga
em RAM). No Llama-3-8B, embedding e saída são 13% dos parâmetros: um "2 bits" de artigo vira ~3,8 bits
no arquivo se esses dois ficam em FP16 [conta, 12 §1].

---

## 3. Contradições entre as pesquisas

Cada item diz o que um documento afirma, o que outro afirma, e como fica.

### K1. "Fronteira" quer dizer duas coisas

- **Doc 14 §3 e `docs/10`** chamam a curva dos k-quants com imatrix de "fronteira".
- **Doc 12 §4** mostra que, no mesmo gráfico e no mesmo protocolo, o EXL3 a 4,0 bits tem KL 0,013,
  contra 0,018 do Q4_K_M a ~4,8 bits [fato, gráfico do exllamav3]. No mesmo KL, o trellis gasta 1,0 b
  a menos no tamanho do Q4_K_M e 1,56 b a menos no do Q6_K [conta, 12 §4].
- **Teoria (doc 12 §5):** quantizador escalar fica 0,46 b acima do limite numa gaussiana a 2 bits;
  o trellis do QTIP, 0,07 b [conta sobre a tabela 1 do QTIP].
- **Como fica:** são duas réguas. Para quem roda llama.cpp oficial na CPU, os k-quants são a
  **fronteira prática**. Para qualidade por bit, o adversário é o trellis. Os textos do STRATA devem
  dizer "fronteira prática do llama.cpp".
- **O que ninguém mediu:** se o trellis mantém essa vantagem na CPU sem perder velocidade. O
  ik_llama.cpp tem `IQ*_KT` na CPU, mas não achamos tabela controlada [fato, 12 G2]. Na GPU, o
  IQ2_KT gerou 146–194 tok/s contra 208 do IQ2_XXS numa RTX 4080 [fato, 12 §7], ou seja, o trellis
  paga 7–30% de decodificação na GPU com o mesmo tamanho [conta].

**Quanto vale 1 bit, se ele existir na CPU [conta]:** no 8B, 1 b por peso no decoder são
6,98 × 10⁹ / 8 ≈ 0,87 GB. O Q4_K_M passa de ~4,9 para ~4,0 GB. Numa máquina que lê a 80% da banda,
isso daria até ~1,2× de velocidade, menos o custo de decodificar o trellis, que na CPU não foi medido.

### K2. "Cada bit a menos é velocidade na mesma proporção"

- **Doc 14 §1** diz que, nas máquinas populares, os bits viram velocidade "na mesma proporção".
- **Doc 15 §5** diz que ler 2 bits em vez de 4 na falta de cache daria "até ~2×".
- **Doc 13 §2.2** mostra que, quanto menos bits e mais banda, mais longe do teto: Q4_0 no M2 Ultra
  usa só 45% da banda, contra 69% do F16 [fato, discussão #4167 do llama.cpp].
- **Doc 14 §2 (flash-moe):** o Qwen3.5-397B passou de 209 GB (4 bits) para 120 GB (2 bits), 1,74× menos
  bytes, e a velocidade subiu de 4,36 para 5,74 tok/s, 1,32× [fato]. A elasticidade medida é
  ~0,5: cada 10% a menos de bytes deu ~5% de velocidade [conta: ln 1,32 / ln 1,74].
- **Doc 14 §2 (PR #25294 do llama.cpp):** o mesmo tipo de modelo, lendo o SSD com cache próprio e E/S
  assíncrona em vez do `mmap`, passou de 0,87 para 2,20 tok/s, 2,5×, **sem mudar um bit** [fato].
  O `mmap` em páginas de 4 KB tirou 377 MB/s de um NVMe que lê 2,98 GB/s em sequência [relato, 15 §3].
- **Como fica:** "bits = velocidade" vale bem para modelo denso que cabe na memória de uma máquina
  lenta. No SSD, o padrão de leitura vem antes dos bits. Para o STRATA, a peça mais útil no regime
  de SSD é o layout (planos por tensor, um arquivo por estrato, índice com offsets), não o número de
  bits em si [hipótese].

### K3. A especulativa com resíduo promete mais do que a própria conta dá

- **Doc 13 L1** apresenta "saída idêntica ao BF16 numa placa de 8–24 GB com velocidade utilizável" e
  mostra 14,9 tok/s no 8B.
- **A mesma tabela do doc 13** mostra que isso é 1,75× o `-ngl` parcial e **3,7× mais lento** que só
  rodar o Q4_K_M na VRAM (55,5 tok/s).
- **Doc 14 L1** chega à mesma ideia e avisa: na fronteira GPU/RAM, "provavelmente empata".
- **Três pontos que nenhum dos dois juntou:**
  1. *O fecho não tem decodificador rápido.* Para sair BF16 exato, os estratos 1–4 incluem o fecho
     em rANS. O doc 13 pede ≥ 25 GB/s de entrada. O STRATA mediu ~65–80 MB/s de saída no formato
     0.2 com 4 núcleos [fato, `docs/05`] e ~135 MB/s de saída no `junta` do 8B com 12 núcleos
     [conta: 16,06 GB / 119 s de decodificação, `docs/05` §31]. Faltam duas ordens de grandeza.
  2. *O valor de "igual ao BF16" é pequeno quando o Q6_K cabe.* No 8B, o STRATA mediu KL × 1000 de
     25,32 (Q4_K_M), 7,55 (Q5_K_M) e 3,31 (Q6_K) contra o F16 [fato, PR #10]. Quem tem 8 GB de VRAM
     roda o Q6_K inteiro (6,6 GB) na placa.
  3. *O α = 0,886 é do 1B, nível 0 contra nível 2* [fato, `docs/08` E1], não do 8B nem contra o
     BF16. Pela regra do projeto, o 1B não decide qualidade (ele já prometeu 3× demais).
- **Como fica:** veja a conta completa na seção 4.2. L1 cai de primeira para o fim da lista.

### K4. Especulativa e MoE não se somam de graça

- **Doc 13 §5 (SpecExec):** com offload, conferir 1 ou 1000 tokens custa o mesmo, porque o tempo é
  a cópia dos pesos [fato].
- **Doc 15 §2:** num MoE, cada token lê só os seus especialistas.
- **Conflito [conta, supondo escolhas independentes e uniformes]:** no Qwen3-30B-A3B (128
  especialistas, 8 por token), conferir 6 tokens de uma vez toca em média
  128 × (1 − (1 − 8/128)⁶) ≈ 41 especialistas por camada, contra 8 de um token só. São ~5× mais bytes
  na conferência. O roteamento real é correlacionado e deve tocar menos [hipótese], mas a premissa do
  SpecExec ("conferir mais tokens sai de graça") não vale em MoE.
- **Como fica:** a L1 do doc 13/14 (especulativa) e a L1 do doc 15 (cache por plano) são ideias
  separadas. Juntar as duas precisa de uma medida própria; não dá para multiplicar os ganhos.

### K5. Sem perda: "pare de investir" contra "sem perda também é velocidade"

- **Doc 12 §6 e §9.4:** tamanho sem perda está saturado (ZipNN 66,4%, DFloat11 ~68–70%, STRATA 0.2
  66,2%); a folga é ~0,07 b [fato, `docs/05` §12]. Recomenda parar de investir em tamanho.
- **Doc 13 L4 e doc 15 §4.5:** o DFloat11 gera 2,3–46× mais que mandar o BF16 cru para a CPU
  [fato]. Sem perda vira velocidade quando o gargalo é mover bytes.
- **Como fica:** as duas coisas são verdade. O tamanho acabou; o que não foi explorado é **onde**
  decodificar (no trecho lento, PCIe ou SSD). Mas isso só serve a quem roda BF16 ou FP8. No Q4_K o
  teto é ~3%, porque os índices já têm 3,86 de 4 bits de entropia [fato, `docs/05`]. E o decodificador
  do STRATA é de CPU e lento (K3, ponto 1). Para o usuário de Q4, não há ganho aqui.

### K6. O alvo de hardware não é o mesmo nos quatro textos

- **Plano do MVP:** notebook x86 de 8 GB de RAM, 4 núcleos, **sem GPU** (memória do projeto,
  2026-10-26).
- **Doc 13 L1:** exemplo numa placa de 8 GB com PCIe. **Doc 12:** o melhor formato é CUDA. **Doc 15:**
  os casos fortes são Mac de 48 GB, PC de 32–64 GB com SSD rápido, ou servidor de 1 TB.
- **Steam, set/2026 [fato, 13 §3]:** 54% dos PCs têm 8 ou 16 GB de VRAM, 80% têm 16 ou 32 GB de RAM,
  6,6% têm 48 GB ou mais.
- **Como fica:** o projeto precisa escolher uma máquina-alvo por pergunta. Para o MVP (CPU de 8 GB),
  a única alavanca das pesquisas que se aplica é menos bits por peso no mesmo KL, ou seja, o motor de
  compressão e o trellis na CPU (K1). Especulativa por PCIe e cache de especialistas não se aplicam
  a essa máquina.

### K7. Qualidade por bit contra compatibilidade (visão c)

- **Visão (c):** converter direto dos formatos de hoje. O ponto forte do formato 0.3 é exatamente esse:
  o estrato 0 é o Q4_K_M do Hub byte a byte [fato, `docs/05` §31].
- **Doc 12:** o caminho para ganhar ~1 bit é o trellis, que só existe no EXL3 (formato próprio, CUDA)
  e no fork ik_llama.cpp. O livro de códigos (`docs/09`) já tinha tirado o Q4_K_M do Hub do nível 0.
- **Como fica:** ir para o trellis troca compatibilidade por qualidade. Um caminho que guarda as duas
  coisas [hipótese]: manter o Q4_K_M do Hub como estrato 0 para baixar, e montar na carga o formato
  de execução (é a lacuna G6 do doc 12, "guardar diferente de rodar"). O custo de transcodificar na
  carga não foi medido por ninguém.

### K8. Números de fonte que não batem entre si

- KTransformers no DeepSeek-V3: a manchete do tutorial diz 11,26 tok/s, a tabela da mesma página diz
  13,69 [fato, 15 §8]. O doc 14 usa 13,69 sem a ressalva. Os dois documentos devem citar a tabela e
  a ressalva.
- Banda do Apple M1: 79% do teto no doc 13, 80% no doc 14. É a mesma medida com arredondamento
  diferente (14,15 tok/s). Sem conflito.

---

## 4. Combinações reais

Combinação = duas peças que existem em trabalhos separados, mais a peça que o STRATA já tem. Para
cada uma: a conta, e o que a mataria.

### 4.1 Trellis na CPU como régua e como alavanca do MVP (docs 12 + 14)

- **Peças:** doc 14 diz que, na máquina popular, bytes viram velocidade quase 1 para 1 (77–85% da
  banda) [fato]. Doc 12 diz que o trellis gasta ~1 b a menos no tamanho do Q4_K_M [conta].
- **Conta:** no 8B, ~0,87 GB a menos de RAM e até ~1,2× de velocidade, se a decodificação do trellis na
  CPU custar pouco. Na GPU ela custa 7–30% (K1).
- **O que o STRATA tem:** a régua (scripts de KL contra F16, `llama-bench`), os níveis como tipos do
  ggml, e o C2.
- **Mata se:** o IQ4_KT na CPU não ficar ≥ 0,3 b à frente dos k-quants no mesmo KL, ou ficar abaixo
  de 0,90× a velocidade do Q4_K_M (régua do N2, doc 12 §10).
- **Bloqueio:** compilar e rodar código externo no contêiner foi recusado na sessão do doc 12. Precisa
  de autorização do Cantyll.

### 4.2 Especulativa com resíduo: a conta completa (docs 13 + 14)

Mesma fórmula de `docs/08` e do doc 13 (tokens por rodada = (1 − α^(γ+1)) / (1 − α), custo =
γ·c + 1). Uso α = 0,886, que é otimista (K3, ponto 3). Eficiência de 80% da banda em tudo [conta].

**Caso A, fronteira GPU/RAM: 14B numa RTX 3060 de 12 GB (360 GB/s), PCIe 4.0 a 25 GB/s.** O Q4_K_M
(9,0 GB) cabe, o Q6_K (12,1 GB) não.

| estratégia | qualidade (KL × 1000 no 8B, como referência) | tok/s [conta] |
|---|---:|---:|
| só Q4_K_M na VRAM | 25,32 | ~33 |
| Q6_K com `-ngl` parcial, 1,6 GB na CPU (DDR4-3200 / DDR5-5600) | 3,31 | ~13 / ~17 |
| especulativa: rascunho Q4_K_M na VRAM, conferência lendo +1,71 b/peso (3,1 GB) pela PCIe | 3,31 | ~15 (γ = 6) |

A especulativa empata com o `-ngl`, que já existe. As duas ficam com metade da velocidade do Q4.
Trocar pelo Q5_K_M (KL 7,55, ~10,6 GB, cabe apertado) deve dar ~27 tok/s [conta], o que domina as duas.

**Caso B, fronteira RAM/SSD: 70B denso num PC de 64 GB sem GPU (DDR5-5600, NVMe Gen4 a 7 GB/s).**

| estratégia | tok/s [conta] |
|---|---:|
| só Q4_K_M na RAM (42,8 GB) | ~1,7 |
| especulativa: rascunho Q4_K_M na RAM, conferência lendo +1,71 b/peso (15,1 GB) do SSD | ~0,7 (γ = 6) |

Mesmo com tudo no teto, nenhum caso de modelo denso chega a conversa com qualidade de Q6.

**Conclusão [conta + hipótese]:** a especulativa com resíduo só ganha quando o rascunho é muito menor
que a conferência e a conferência é muito lenta. Quando isso acontece, a velocidade absoluta já é
baixa. O teto é ~1/(1 − α) ≈ 8,8 tokens por conferência. A L1 sai do topo da lista. Ela só volta se
aparecer um rascunho bem menor que o Q4 (2–3 bits) com α alto, o que leva à combinação 4.4.

### 4.3 Cache por plano de especialista com leitura alinhada no SSD (doc 15 L1 + L2, doc 14 L2)

- **Peças:** HOBBIT mostra que trazer a falta em menos bits custa ≤ 1% de acurácia, mas guarda cópias
  separadas [fato]. DPA-MoE mostra planos aninhados com 4–8% de custo de kernel, tudo residente na
  GPU [fato]. O PR #25294 e o flash-moe mostram que o SSD só serve com leitura própria e cache [fato].
- **O que o STRATA tem:** planos por tensor em streams separados, troca a quente bit a bit em ≤ 16 µs
  (`docs/08` E3), SHA-256 por plano. Nos safetensors do HF, cada especialista é um tensor 2-D, e um
  plano de 1 bit por matriz tem 197 KB no Qwen3-30B-A3B [conta, 15 §5], acima dos 32 KiB que o LLM in
  a Flash recomenda.
- **Onde a vantagem sobre o HOBBIT realmente mora [conta]:** numa falta fria, os dois leem a base de
  2 bits: mesmos bytes. A diferença aparece só quando um especialista **sobe** de precisão: o HOBBIT lê a
  cópia de 4 bits inteira (~4,5 b/peso), o cache por plano lê só os planos que faltam (~2,25 b/peso).
  O ganho total depende da fração do tráfego que é subida.
- **Pré-teste novo, mais barato que o M1:** gravar o roteamento do OLMoE-1B-7B no contêiner e contar,
  numa simulação de cache, quanto do tráfego é falta fria e quanto é subida. Se a subida for < 30% dos
  bytes, o ganho máximo sobre o HOBBIT fica abaixo de ~15% e o M1 nem precisa rodar [conta: metade
  de 30%].
- **Bloqueios:** o modo residual 0.3 só aceita BF16 2-D; GGUF guarda especialistas em 3-D; os tipos D e
  livro não têm `MUL_MAT_ID` [fato, 15 §5]. E a frente MoE está fechada até o Cantyll pedir.

### 4.4 Trellis aninhado como base pequena para leitura parcial (doc 12 G1 + docs 13–15)

- **Peças:** a gaussiana é refinável sem perda no limite (Equitz e Cover, 1991) e o refinamento
  sucessivo de trellis existe na teoria desde 1999 [fato, 12 G1]. O Any-Precision aninha um escalar não
  uniforme quase de graça [fato]. O STRATA mediu ~0,18 b por nível para aninhar o livro no 8B [fato,
  `docs/09`].
- **Por que é a combinação que destrava as outras [hipótese]:** a especulativa (4.2) e o cache por
  plano (4.3) precisam de uma base pequena e boa (2–3 bits) e de planos baratos por cima. Uma base
  escalar a 2–3 bits é exatamente onde os escalares mais perdem (IQ2_M precisa de 0,54 b a mais que
  o EXL3 no mesmo KL [conta, 12 §4]).
- **Mata se:** o N1 medir que o aninhamento do trellis custa > 0,20 b por nível numa gaussiana (aí
  a vantagem de ~0,4 b sobre o escalar some).
- **Custo:** 1 a 2 dias, só numpy, no contêiner [doc 12 §10].

### 4.5 Armazenamento: Xet, B.10 e receita no mesmo conjunto (doc 13 L2 + `docs/05` §31–32)

- **Peças:** o Xet guardou 29 quantizações do gemma-2-9b-it-GGUF (191 GB) em ~97 GB num ambiente de
  teste [fato, 13 L2]. O B.10 guarda BF16 + Q4_K_M + Q5_K_M + Q6_K do 8B em 11,18 GB em vez de
  33,31 (−66%) [fato]. A receita (BF16 + imatrix + llama.cpp b3821) recriou os 17 GGUFs do bartowski
  do 1B byte a byte, com FMA [fato, `docs/05` §32].
- **Conta:** o conjunto do 8B pela receita = BF16 sem perda no formato 0.2 (~66% de 16,06 ≈ 10,6 GB)
  + imatrix (alguns MB), e cobre os 17 GGUFs, não só 3. Fica abaixo dos 11,18 GB do B.10. O 8B pela
  receita ainda não foi conferido (precisa do b3472).
- **Hipótese sobre o Xet nesse conjunto:** como Q4_K, Q5_K e Q6_K da mesma matriz têm bytes diferentes,
  a deduplicação por pedaços só deve achar os tensores que ficam iguais entre os arquivos (o
  `output.weight` em Q6_K nos três, as normas). No 8B, o `output.weight` em Q6_K tem ~0,43 GB; contar
  ele uma vez só poupa ~0,86 GB de 33,31 GB, ~3% [conta]. Se houver outros tensores iguais, um pouco
  mais. Não conferi se o Xet também comprime os pedaços.
- **Mata o argumento de armazenamento do B.10 se:** o Xet sozinho chegar perto de −66% no mesmo
  conjunto. A receita já é um rival mais forte que o Xet para quem hospeda. O que sobra de próprio do
  STRATA é o lado de quem baixa: quem já tem o Q4_K_M baixa 6,25 GB em vez de 16,06 para ter o BF16
  exato [fato, `docs/05` §31].

### 4.6 Esparsidade de ativação com planos de resíduo (doc 14 L3)

- **Peças:** TEAL zera 40–50% das ativações sem treino, com 1,53–1,8× de velocidade [fato]. Os planos do
  STRATA podem ser lidos por linha.
- **Conta:** se metade das linhas está desligada, o resíduo custa metade dos bytes.
- **Mata se:** os blocos de 256 pesos dos k-quants forem grandes demais para pular linhas, ou se a KL de
  "nível 2 só nas linhas ativas" ficar perto da KL do nível 0 inteiro. Teste em PyTorch no 1B, horas.

---

## 5. Lacunas ranqueadas

Escalas de 1 a 5. **Valor:** quanto muda a visão do projeto se der certo. **Custo:** 1 = horas no
contêiner, 5 = GPU grande ou semanas. **Risco:** 1 = quase certo que dá uma resposta útil, 5 = quase
certo que a ideia morre. **Nota** = Valor × (6 − Custo) × (6 − Risco), máximo 125. A nota é uma régua
para discutir, não um cálculo exato; os números de valor e risco são opinião minha [hipótese].

| # | lacuna | de onde vem | valor | custo | risco | nota | bloqueio |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Xet × B.10 × receita no mesmo conjunto (1B, depois 8B) | 13 L2 + `docs/05` §32 | 3 | 1 | 2 | 60 | o Xet oficial é código externo; dá para começar com um chunker próprio em Python |
| 2 | N2: trellis (IQ4_KT) contra k-quants na CPU, 1B | 12 G2 | 4 | 2 | 3 | 48 | autorização para compilar o ik_llama.cpp |
| 3 | N1: trellis aninhado numa gaussiana | 12 G1 | 5 | 2 | 4 | 40 | nenhum |
| 4 | pré-teste MoE: fração de subidas no roteamento do OLMoE-1B-7B | seção 4.3 | 3 | 2 | 4 | 24 | frente MoE fechada |
| 5 | leitura alinhada do SSD por especialista e plano, PC de 32–64 GB | 15 L2 | 4 | 4 | 3 | 24 | MoE fechado; tensores 3-D; `MUL_MAT_ID` |
| 6 | fecho a partir de base trellis (N3) | 12 G3 | 2 | 3 | 3 | 18 | depende do N2 |
| 7 | TEAL × planos de resíduo, 1B | 14 L3 | 2 | 2 | 4 | 16 | nenhum |
| 8 | base comum da camada + deltas por especialista (energia do delta) | 15 L3 | 3 | 2 | 5 | 12 | MoE fechado |
| 9 | decodificador rápido do fecho (GPU ou rANS intercalado) | 13 L4, K3 | 2 | 4 | 3 | 12 | nenhum; só serve a BF16 |
| 10 | especulativa com resíduo: α do Q4_K_M contra Q6_K e BF16 | 13 L1, 14 L1 | 2 | 1 | 5 | 10 | nenhum; a conta da seção 4.2 já indica empate |

Trabalho de mesa, sem experimento, que vale fazer junto com qualquer item acima:

- **Regra dos dois números de bits** (matrizes de camada e arquivo inteiro) em toda tabela (12 G4).
- **Trocar "fronteira" por "fronteira prática do llama.cpp"** nos textos do motor (K1).
- **Custo da RAM em 2026** como argumento (13 L5): o módulo DDR5 de 32 GB passou de ~US$ 95 para
  ~US$ 400 no varejo entre set/2025 e dez/2025, segundo fontes secundárias [fato, 13 §7]. Isso muda o
  argumento de "armazenamento mais barato" para "menos RAM comprada".

### Regras de parada, escritas antes

1. **Xet × B.10 × receita.** Mede, no mesmo conjunto (1B: BF16 + Q4_K_M + Q5_K_M + Q6_K do bartowski),
   o tamanho de: (a) arquivos crus; (b) deduplicação por pedaços de ~64 KB com corte definido pelo
   conteúdo; (c) B.10; (d) receita (BF16 no 0.2 + imatrix). Se (b) ficar a menos de 10 pontos
   percentuais de (c), o B.10 deixa de ser argumento de hospedagem e o texto público passa a falar só
   do lado de quem baixa.
2. **N2.** Régua do doc 12 §10: IQ4_KT ≥ 0,3 b à frente da curva dos k-quants no mesmo KL e ≥ 0,90×
   a velocidade de geração do Q4_K_M na CPU. Passou no 1B: repetir no 8B antes de concluir.
3. **N1.** Régua do doc 12 §10: ≤ 0,10 b por nível passa para o 1B; > 0,20 b mata.
4. **Pré-teste MoE.** Subidas < 30% dos bytes de falta: o M1 não roda e a L1 do doc 15 morre.

---

## 6. O que fica rebaixado, e por quê

- **Especulativa com resíduo (L1 dos docs 13 e 14):** empata ou perde nos dois casos densos da seção
  4.2, depende de um decodificador ~100× mais rápido que o do STRATA para sair BF16 exato, e não se
  soma com MoE (K4).
- **Sem perda para Q4:** teto de ~3% (K5).
- **Kernel mais rápido para modelo que cabe:** o llama.cpp já está em 77–85% da banda nas máquinas
  populares [fato, 14 §1].
- **Mais ajustes escalares no motor depois do C2:** a teoria deixa ~0,4–0,7 b que só quantização
  vetorial ou trellis recupera (12 §5). Recomendo terminar o C2 no 8B (já está pronto para rodar) e,
  depois dele, levar o motor para o N1 em vez de abrir mais variantes escalares [hipótese].

---

## 7. O que as pesquisas dizem sobre cada item da visão

| visão | o que as pesquisas dizem | etiqueta |
|---|---|---|
| (a) armazenar modelo grande mais barato | −66% medido no 8B e no 70B. Sem perda saturado em ~66% para todos os codecs. A receita já ganha do B.10 para quem hospeda. O ganho próprio do STRATA é o lado de quem baixa. | fato + conta |
| (b) rodar modelo grande em máquina popular | Denso: limite de capacidade e banda da RAM; 70B em 16 + 32 GB dá ~2–3 tok/s no teto. O que muda isso é menos bits no mesmo KL (trellis, motor) ou MoE. | fato + conta |
| (c) converter direto dos formatos de hoje | É o ponto mais forte do 0.3 (Q4_K_M do Hub byte a byte). O caminho para ganhar 1 bit (trellis) briga com isso (K7). | fato + hipótese |
| (d) ir além da barreira da RAM | Só foi medido com MoE + SSD + cache (397B a 4,36 tok/s num Mac de 48 GB; 103 GB a 8,8 tok/s com 24 GB de RAM). Para modelo denso, nenhuma rota fecha (AirLLM: dezenas de segundos por token pela conta). O STRATA não tem peça de MoE pronta. | fato + conta |

**Decisão que estas pesquisas levantam (é do Cantyll):** se a visão (d) continua sendo objetivo, ela
passa pela frente de MoE, que hoje está fechada. O primeiro passo seria o pré-teste da seção 4.3
(horas, contêiner), sem tirar o motor da frente.

---

## 8. Plano em ordem

1. **Xet × B.10 × receita no 1B** (contêiner, horas). Decide o argumento de armazenamento.
2. **N2 no 1B**, se o Cantyll autorizar compilar o ik_llama.cpp (contêiner, horas). Decide qual é a
   régua e se existe 1 bit para pegar na CPU.
3. **N1** (contêiner, 1 a 2 dias). Decide se o motor de compressão muda de escalar para trellis.
4. **C2 do 8B**, como já planejado no `docs/10`, quando houver máquina (o caderno do Kaggle está pronto).
5. **Pré-teste MoE**, só se o Cantyll abrir a frente.

Os itens 1 a 3 rodam no contêiner e não competem com o motor por GPU.
