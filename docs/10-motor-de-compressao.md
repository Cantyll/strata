# Motor de compressão: a prioridade única

Plano escrito em 2026-10-26, antes de rodar qualquer medida nova.

## A decisão

Cantyll, 2026-10-26: a prioridade agora é resolver o motor de compressão. A ordem do projeto passa a ser:

1. **Compressor:** qualidade por tamanho dos níveis contra os k-quants com imatrix.
2. **Leitor e remontador:** `junta`, leitura em fluxo, subida de nível.
3. **Executor:** kernels, MVP local, precisão sob demanda.

O que está fora do compressor fica parado até este plano dar um veredito. O livro de códigos já estava congelado ([09](09-livro-de-codigos.md)). O E6 do 8B e o MVP local esperam.

## Onde está o atraso

Llama-3.1-8B, caminho F16, KL ×1000 contra o BF16 em 100 trechos. "Bits a mais" é quanto o nível gasta além do k-quant com imatrix de mesma KL, no mesmo tamanho de RAM ([05, seções 28 e 30](05-resultados.md#30-sub-células-por-tensor-no-llama-31-8b-tipos-d-na-gpu)).

| | nível 1 | nível 2 | geração na GPU |
|---|---:|---:|---|
| k-quant no mesmo tamanho (interpolado) | 7,13 | 3,09 | |
| tipos P (os de hoje, base = Q4_K_M do Hub) | 10,78 (+0,34 b) | 3,66 (+0,20 b) | 1,00 / 1,20× do k-quant |
| tipos D (sub-células por tensor) | 9,27 (+0,23 b) | 3,40 (+0,11 b) | 0,64 / 0,79× (reprova) |
| livro aninhado (estrato 0 deixa de ser o Hub) | +0,006 b (Llama), +0,067 b (Qwen3) | −0,027 b (Llama) | 0,93× (reprova) |

Em KL, o nível 1 D está 30% acima do k-quant e o nível 2 D, 10%. Perto do Q5, um bit a mais por peso divide a KL por ~2,3 (Q5_K_M 7,55 contra Q6_K 3,31, ~1 bit/peso de diferença), então 0,1 bit vale ~8% de KL.

O único tipo que passa na velocidade em todo lugar é o P, e ele é o mais atrasado. Qualquer ganho que precise de kernel novo começa devendo.

## Diagnóstico: por que os níveis perdem

**1. A escala é escolhida para o nível 0.** A imatrix ajusta a escala de cada sub-bloco do Q4_K para que os pesos importantes caiam perto das 16 marcas. Os níveis herdam essa escala. O k-quant de 5 ou 6 bits refaz a busca de escala para a sua própria grade; o nível não pode, porque a base é fixa. Analogia: o k-quant corta cada bolo com a faca do tamanho certo; o nível só pode cortar ao meio as fatias que já estão no prato. As escalas por nível e as sub-células ([05, seções 26 a 30](05-resultados.md#26-níveis-com-escala-própria-o-q4-fica-intacto)) recuperaram parte disso.

**2. O bit do nível não tem liberdade nenhuma.** Com a base fixa e os pontos dos filhos fixos, o erro ponderado pela imatrix é uma soma de termos de um peso cada. Então o melhor bit de cada peso é sempre o do filho mais próximo, e o escritor não tem o que escolher. Todo o esforço até aqui mexeu em **onde ficam os filhos** (escala, deslocamentos, livro), nunca em **qual filho cada peso usa**. A única liberdade que sobra está na interação entre pesos: errar um pouco para cima num peso para compensar o erro de outro que multiplica a mesma ativação. É o que o GPTQ faz com a matriz de covariância das entradas (a "Hessiana" H = XᵀX). A imatrix é só a diagonal dessa matriz.

**3. A distribuição dos bits entre matrizes é herdada.** Os níveis dão +1 bit a toda matriz, na proporção da receita do Q4_K_M. Na mistura do E5 ([08, seção 9](08-precisao-sob-demanda.md#9-resultado-do-e5-llama-31-8b-colab-a100-caminho-f16)), o alocador pelo erro com imatrix deixou para o fim matrizes que pesam muito na saída: a KL cai de 10,5 para 7,5 entre 5,63 e 5,78 GiB. Isso não muda os níveis inteiros, mas é o que faz a mistura perder 0,3 a 0,6 bit/peso no meio da escada.

**Correção de um diagnóstico antigo.** Os ~3,6 bits/peso fixos de sinal e expoente eram o gargalo do formato 0.2 (planos de bits do BF16). O modo residual (0.3) começa no Q4_K_M e não paga esse custo; ele não é o problema de agora.

## O que já foi tentado e não se repete

| ideia | onde | resultado no 8B |
|---|---|---|
| reconstrução fora do centro da célula (ponto médio com viés, deslocamento por tensor ou por linha) | 05 §26–30, tipos D | o melhor até hoje com a base do Hub: +0,23 / +0,11 b. Por linha, as tabelas comem o ganho |
| escala própria por nível ou por bloco | 05 §26 | nível 1 +18% de KL contra o Q5_K_M |
| escala conjunta (busca do Q4 pesando os níveis de cima) | 05 §27–28 | ganha num nível, paga no outro; soma 0,32 b |
| base fina (o Q4 sai por corte da grade fina) | 05 §25, §28 | melhor nível 1 (+0,07 b), paga +0,22 b no nível 0 |
| livro de códigos aninhado | 09 | empate no Llama, cinza no Qwen3, kernel lento; congelado |
| entropia nos planos dos níveis | 06 (decisões) | cada bit já é quase imprevisível: cru custa 1 bit, o modelo dava ~1,04. Só mexeria no disco, não na RAM |
| GPTQ/AutoRound sobre o k-quant | 07 | descartado por render pouco nos k-quants. Ver C1: o caso dos níveis é outro |

Duas das quatro ideias levantadas para este plano (ponto médio/viés e escalas melhores) já são os tipos D e as seções 26–28. A terceira (entropia nos planos) já foi medida e não ataca a RAM. Sobra o alocador, e entra uma nova: escolher os bits com a Hessiana.

## Os candidatos

### C1. Bits do nível escolhidos com a Hessiana (GPTQ nos planos)

**Ideia.** O nível 0 fica igual byte a byte ao Q4_K_M do Hub. Os pontos dos filhos ficam como nos tipos P (ou D). O que muda é só a escolha do bit: em vez do filho mais próximo, o escritor percorre as colunas na ordem do GPTQ e, a cada coluna, passa o erro do arredondamento para as colunas que faltam, pela H. Quando o alvo compensado sai da célula do pai, o bit vai para o filho da ponta (a base não muda).

**Por que pode dar certo onde o AutoRound não deu.** No k-quant, a busca de escala por taxa já pega quase tudo, e o GPTQ chega tarde. No nível, a escala está presa e o bit não tem nenhuma outra liberdade (diagnóstico 2). É a única alavanca que nenhum experimento tocou.

**Por que pode não dar.** O bit só escolhe entre dois vizinhos a meio passo um do outro, então a compensação é pequena. A literatura mostra ganho do GPTQ sobre arredondamento simples grande em 3 e 4 bits e menor em 6 e 8. Não sei estimar o tamanho do ganho aqui sem medir.

**O que o resultado custa ao formato:** nada. Os planos continuam 1 bit por peso, os tipos P rodam como estão, sem kernel novo. Se der certo nos tipos P, o atraso cai sem perder a velocidade.

**Riscos que a régua precisa pegar.**

- Ajuste ao texto de calibração: o GPTQ usa a calibração com mais força que a imatrix. A KL é medida também num segundo texto, fora do wikitext.
- O nível 2 é escolhido depois do nível 1. Os bits do nível 1 ficam presos ao que o GPTQ escolheu nele.

**Régua (escrita antes de rodar).** Eixo e frente como no [09](09-livro-de-codigos.md#régua-de-decisão-definida-antes-de-rodar): KL ×1000 no caminho F16, contra a frente dos k-quants com imatrix em F16 (Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0), em bits por peso no mesmo tamanho de RAM.

| etapa | passa | descarta |
|---|---|---|
| 1B, contêiner (triagem) | nível 1 com GPTQ tira ≥ 10% da KL dos tipos P e o nível 2, ≥ 5% | abaixo disso nos dois níveis: para aqui, sem 8B |
| 8B, A100 (decide) | **sucesso:** tipos P com GPTQ em ≤ +0,05 b nos níveis 1 e 2, nível 0 igual ao Hub, e o ganho sobre os P de hoje mantém pelo menos metade no segundo texto | **zona cinza:** algum nível entre +0,05 e +0,12 b, ou só os D chegam lá (D reprova na velocidade). **Fracasso:** nível 1 acima de +0,12 b com P e com D |

O 1B só serve para descartar. Ele já prometeu mais do que o 8B entregou pelo menos quatro vezes.

**Resultado no 1B (2026-10-27, [tabela](../tests/kernel/resultados/unsloth__Llama-3.2-1B_gptq_container.md)): passa na triagem.** Com calibração misturada (metade wikitext, metade C4), amortecimento 0,1 e Hessianas de 500 trechos, o nível 1 dos tipos P vai de +0,52 para −0,11 bit no wikitext e de +0,44 para +0,04 no C4; o nível 2 tira 31% da KL no wikitext e 16% no C4. Calibrar só com wikitext fez o GPTQ decorar o domínio (nível 2 pior que o de hoje no C4); a mistura corrigiu, e 4× mais tokens nas Hessianas levaram a retenção do nível 2 no C4 de 18% para 49%. Os pontos D quase não somam nada sobre o GPTQ (medido com a calibração só wikitext), então o tipo P basta. A régua do 8B pede 50%: o nível 2 está no limite.

### C2. Alocador pelo efeito medido na saída

**Ideia.** No lugar do erro com imatrix por matriz, medir no 8B quanto a KL cai ao subir cada grupo de matrizes um nível (por exemplo, tipo de matriz × faixa de 4 camadas: 7 × 8 = 56 grupos, dois degraus cada). Depois, mochila com os ganhos medidos. Salvar o plano por matriz no JSON.

**O que pode ganhar.** Só o meio da escada (5,33 a 5,63 GiB, hoje +0,29 a +0,59 b). Não pode deixar nenhum ponto melhor que os níveis inteiros, que são o teto.

**Régua.** Sucesso: no 8B, de 5,33 a 5,63 GiB, a mistura fica ≤ +0,10 b além do atraso dos próprios níveis, e nenhum orçamento piora contra o alocador de hoje. A soma dos ganhos medidos tem de prever a KL de três orçamentos com erro ≤ 10%; se não prever, os efeitos não somam e a mochila não vale.

**Depende do C1:** os ganhos por grupo mudam se os bits mudarem. Rodar depois.

### Fora do plano

- **Entropia nos planos:** já medida (06). Não ataca a RAM.
- **Descongelar o livro:** é a única ideia que empatou no Llama, mas troca a base do Hub, precisa de tipo novo no llama.cpp e ficou cinza no Qwen3. É candidato natural se o C1 fracassar, já que formato e kernel podem mudar; o kernel tem de passar a fusão gate+up e a velocidade.

## Ordem e custo

| # | experimento | onde | trabalho | máquina |
|---|---|---|---|---|
| 1 | C1 no 1B: H por camada com 125 trechos do texto de treino, GPTQ nos planos P e D, KL em 40 trechos | contêiner (CPU, 4 núcleos) | ~1 dia de código; ~3–4 h de CPU | grátis |
| 2 | C1 no 8B: o mesmo, H camada por camada, KL em 100 trechos + segundo texto, frente k-quant em F16 | Colab A100 High-RAM | notebook autossuficiente | ~2 h de A100 |
| 3 | C2 no 8B, sobre os níveis que saírem do C1 | Colab A100 | notebook | ~1,5 h de A100 |

Só o passo 1 roda antes de outro sinal verde. Os passos 2 e 3 só saem se o anterior passar na régua.

**Sem regra de parada (Cantyll, 2026-10-26).** O motor só sai da prioridade quando estiver resolvido. Mudar o formato, o kernel ou o próprio jeito de empilhar os estratos vale, se for isso que resolve. Se o C1 fracassar, o próximo passo não é outra variação de escala ou de tabela, que já renderam o que tinham: desde o PR #9, dez rodadas baixaram a soma do atraso (níveis 1 + 2) de 0,54 para 0,31–0,34 bit/peso, e cada uma rendeu menos que a anterior. O próximo passo é soltar uma restrição de estrutura: a base igual ao Hub (o livro, ou a base escolhida pelos níveis de cima), o 1 bit cru por nível, ou o próprio aninhamento. Cada uma entra com régua própria antes de rodar.

## O que um sucesso permite dizer

- **Pode:** "os níveis 1 e 2, sobre o Q4_K_M do Hub sem mudar um byte, ficam a menos de 0,05 bit/peso dos k-quants com imatrix publicados, rodando no kernel de hoje".
- **Não pode:** "o STRATA quantiza melhor que o k-quant". Um k-quant feito com GPTQ também melhoraria. A vitória seria contra o que se publica, não contra a técnica.
