# STRATA

**Um formato de arquivo para pesos de IA que guarda só o que não dá para prever.** Cada estrato refina o anterior, cada modelo-filho guarda só a diferença para o pai, e o arquivo volta bit a bit para o `.safetensors` original.

![Cortar o .strata em qualquer estrato devolve um modelo válido](docs/img/diagrama-estrutura.png)

## A ideia em três linhas

- **Um arquivo, várias precisões.** O estrato 0 é o GGUF Q4_K_M publicado no Hub, sem mudar um byte. Cada nível seguinte acrescenta 1 bit por peso e divide ao meio a célula do Q4 (nível 1 no tamanho do Q5_K_M, nível 2 no do Q6_K), e o último estrato, o fecho, devolve o BF16 exato. Hoje, cada precisão é um arquivo separado.
- **Uma família, não cópias.** Um fine-tune aponta o modelo-base e guarda só a diferença (medido, ainda não gravado no arquivo).
- **Sem perdas por padrão.** O arquivo volta byte a byte para o `.safetensors`, conferido pelo SHA-256.

Esse é o **modo residual** (`strata/0.3`), o modo principal do projeto. O primeiro formato (`strata/0.2`, os planos de bits do BF16: sinal, expoente e mantissa) continua no código e no leitor. Sem perdas ele é um pouco menor (10,59 contra 11,12 bits/peso no Llama-3.2-1B), mas nos cortes com perda perde para os k-quants em todo tamanho e deixou de ser desenvolvido para isso.

O ganho grande não vem de espremer um modelo (sem perdas, o teto fica perto de 30%), e sim de parar de guardar o mesmo modelo várias vezes: um único repositório do Hugging Face guarda 29 versões do mesmo modelo de 9 bilhões de pesos, somando 191 GB.

## O que foi medido em modelos reais

| | Resultado |
| --- | --- |
| Llama-3.1-8B-Instruct, sobre o Q4_K_M do bartowski (imatrix) | 11,13 bits/peso: 11,18 GB no lugar de BF16 + Q4_K_M + Q5_K_M + Q6_K (33,31 GB, −66%); os 4 shards voltam com o SHA-256 do Hub |
| Llama-3.1-70B-Instruct (Colab) | 98,27 GB contra 291,47 GB (−66%); os 30 shards voltam com o SHA-256 do Hub, mas a volta ao BF16 levou 1,7 h em 8 núcleos |
| Níveis contra os k-quants com imatrix (8B, mesmo tamanho de RAM, bits escolhidos com GPTQ) | bits/peso a mais, wikitext / C4: nível 1 −0,02 / **+0,12**; nível 2 (tabela linha) −0,025 / +0,035 |
| Geração na GPU (8B, A100, patch do llama.cpp) | nível 1 a 0,97 do Q5_K_M; nível 2 a 1,17 do Q6_K |

O que estes números não dizem:

- **O −66% vale para quem guarda as quatro versões.** Para quem só quer o BF16, o `strata/0.2` sozinho é ~5% menor. O ganho vem de servir o Q4_K_M e o BF16 com os mesmos bytes.
- **Os níveis ainda não ganham dos k-quants.** No texto fora do domínio da calibração (C4), o nível 1 gasta 0,12 bit/peso a mais que o Q5_K_M para a mesma qualidade. Os bits GPTQ saem de um script Python ([`tests/kernel/strata_gptq_niveis.py`](tests/kernel/strata_gptq_niveis.py)); o escritor `distribui` ainda grava o filho mais próximo, que fica +0,38 / +0,16 bit atrás nos níveis 1 e 2 (wikitext). O efeito dos bits GPTQ no tamanho do fecho não foi medido.
- **Os níveis só rodam no llama.cpp com o patch de [`kernel/`](kernel/).** A qualidade dos níveis com GPTQ foi medida num modelo só, o Llama-3.1-8B.

Detalhes em [docs/05-resultados.md](docs/05-resultados.md) (seções 21 a 34), [docs/10](docs/10-motor-de-compressao.md) e [docs/11](docs/11-kernel-do-tipo-d.md).

## O que a página mostra (modelo sintético, `strata/0.2`)

Medido num modelo sintético BF16 de 1.000.192 pesos (semente 7), com o código deste repositório:

| | Resultado |
| --- | --- |
| Volta para `.safetensors` | SHA-256 idêntico, byte a byte |
| Tamanho do `.strata` | 70,0% do original (o teto teórico fica perto de 67%) |
| Lendo só 46,5% do arquivo (E8M1) | erro de 10,95%, contra 14,48% de um INT4 por bloco |
| Fine-tune guardado como diferença do pai | 24,7% do tamanho do filho guardado sozinho |
| Informação no expoente | 2,70 bits dos 8 que ele ocupa |

Os pesos são sintéticos (caudas pesadas e canais atípicos, como num LLM), então os números servem para comparar opções, não para prometer ganhos. Tabelas completas, outras sementes e um modelo 4× maior em [docs/05-resultados.md](docs/05-resultados.md).

## Experimente

**A página.** [`dist/strata-lab.html`](dist/strata-lab.html) é um arquivo só, sem servidor. Baixe e abra no navegador: ela gera um modelo de teste no aparelho, converte para `.strata`, volta para `.safetensors` e confere o SHA-256, pintando o resultado num painel de azulejos. Nada sai do aparelho.

<p>
  <img src="docs/img/prototipo-celular.png" alt="Primeira tela no celular: O arquivo volta idêntico" width="260">
  <img src="docs/img/prototipo-conversao.png" alt="Conversão em andamento, pintada demão a demão" width="260">
</p>

**Os testes** (Node 18 ou mais novo):

```
node tests/test-lab.mjs     # o pipeline da página: tabela por estrato e seis conferências
node tests/test-core.mjs    # o codec: modelos de 1 M e 4 M pesos, prefixos e bordas de tipos
node tests/test-rans.mjs    # o rANS do expoente; com um .safetensors no fim, mede e confere um modelo real
node tests/test-seguranca.mjs   # o leitor contra arquivos hostis, os hashes e os vetores públicos
node tests/fuzz-leitor.mjs      # fuzzing do leitor; com --rust, compara com o leitor em Rust caso a caso
node tests/test-shards.mjs      # modelos em vários .safetensors: um .strata por shard ou arquivos por estrato
python3 build.py            # monta dist/strata-lab.html a partir de src/
```

**O leitor de referência** (Rust 1.87 ou mais novo) valida, decodifica e confere um `.strata`, ou reconstrói o `.safetensors`:

```
cd leitor-rs && cargo test --release && cargo build --release
./target/release/strata-ref confere modelo.strata
./target/release/strata-ref volta modelo.strata modelo.safetensors
./target/release/strata-ref fluxo modelo.strata    # lê em pedaços, como num download
./target/release/strata-ref residual modelo.safetensors Q4_K_M.gguf modelo.strata   # modo residual (0.3)
./target/release/strata-ref gguf modelo.strata Q4_K_M.gguf   # o GGUF de volta, só do estrato 0
./target/release/strata-ref distribui Q4_K_M.gguf pasta model-0000*.safetensors --index model.safetensors.index.json   # níveis e fecho ao lado de um GGUF publicado
./target/release/strata-ref junta pasta/model.strata.index.json saida --base Q4_K_M.gguf   # os shards originais de volta
```

**Em Python** o mesmo leitor é usado como o safetensors (Python 3.11 ou mais novo; precisa do Rust para compilar):

```
cd leitor-py && pip install maturin && maturin develop --release
python -m pytest tests -q
```

```python
import strata_ref
pesos = strata_ref.load_file("modelo.strata", framework="pt")    # dict nome -> torch.Tensor, conferido
with strata_ref.safe_open("modelo.strata", framework="numpy") as f:
    w = f.get_tensor("model.norm.weight", prof=3)                 # só 3 planos de mantissa
strata_ref.abre_cabecalho("modelo.strata").bytes_ate(3)          # quantos bytes baixar para ler com 3 planos
pesos = strata_ref.load_url("https://.../modelo.strata", framework="pt", prof=3)  # decodifica enquanto baixa
```

Dá para rodar tudo pelo navegador, sem instalar nada no computador: abra o repositório num Codespace do GitHub e use o terminal dele.

## Documentação

| Documento | Para quem quer |
| --- | --- |
| [01 · Pesquisa](docs/01-pesquisa.md) | entender o problema, o estado da arte e a proposta |
| [02 · Especificação do formato](docs/02-especificacao-formato.md) | ler ou escrever um `.strata`: layout binário, cabeçalho, prefixos, exemplo byte a byte e extensões propostas |
| [03 · Fórmulas](docs/03-formulas.md) | conferir a matemática: regra da cadeia, custo por estrato, ponto médio, métricas, genealogia |
| [04 · Protótipo](docs/04-prototipo.md) | saber como a página funciona e como foi desenhada |
| [05 · Resultados](docs/05-resultados.md) | ver todos os números medidos e o que eles não provam |
| [06 · Histórico](docs/06-historico.md) | acompanhar como tudo foi feito, as revisões e as decisões |
| [07 · Quantização aninhada](docs/07-quantizacao-aninhada.md) | o que outros trabalhos fazem com o conflito entre o nível de baixo e os de cima |
| [08 · Precisão sob demanda](docs/08-precisao-sob-demanda.md) | o plano para rodar a partir das camadas: começar no Q4_K_M, subir e descer de nível com o modelo rodando, experimentos e critérios de corte |
| [09 · Livro de códigos](docs/09-livro-de-codigos.md) | o livro de códigos aninhado: empate no Llama, kernel lento, congelado |
| [10 · Motor de compressão](docs/10-motor-de-compressao.md) | a prioridade atual: por que os níveis perdem para os k-quants com imatrix, o que já foi tentado e a ordem dos próximos experimentos com régua |
| [11 · Kernel do tipo D](docs/11-kernel-do-tipo-d.md) | o kernel CUDA dos níveis: por que o D exato reprova na velocidade e a tabela linha passa no nível 2 |
| [DESIGN.md](DESIGN.md) | o sistema visual do painel de azulejos |
| [video/README.md](video/README.md) | o padrão dos vídeos explicativos (9:16, narração, transições 3D) |
| [PRODUCT.md](PRODUCT.md) | o contexto de produto do protótipo |

## O que tem aqui

```
docs/                     pesquisa, especificação, fórmulas, protótipo, resultados, histórico
docs/img/                 diagrama, gráfico e capturas do protótipo
src/strata-core.js        codec: separa os campos, monta e lê o .strata (validado e com hashes), mede o erro
src/strata-lab.js         pipeline da demonstração (roda num Web Worker)
src/strata-pincel.js      pinta na hora os azulejos, frisos, molduras e botões da página
src/strata-lab.src.html   a página, antes da montagem
dist/strata-lab.html      a página montada, pronta para abrir (≈153 KB)
build.py                  injeta os scripts na página e gera dist/
tests/                    testes do codec, do pipeline completo e da segurança do leitor
tests/vetores/            vetores públicos: .strata válidos e hostis com o resultado esperado
leitor-rs/                leitor de referência em Rust (biblioteca e linha de comando strata-ref)
leitor-py/                o leitor de referência em Python (binding do Rust, API como a do safetensors)
kernel/                   tipos dos níveis do modo residual: referência em C, AVX2, CUDA e o patch do llama.cpp
tests/kernel/             escritores dos níveis (GPTQ, tabelas), notebooks do Colab e resultados no 8B
tests/dist/               distribuição sobre o GGUF do Hub e o teste do 70B
video/                    padrão dos vídeos explicativos: regras, exemplo e scripts de render
DESIGN.md, .impeccable/   sistema de design e contrato de direção
PRODUCT.md                contexto de produto
```

## Estado

- **Implementado (`strata/0.3`, modo residual, o principal):** o estrato 0 é o GGUF Q4_K_M byte a byte, os níveis seguintes refinam a célula de cada peso e o último fecha o BF16 exato. No Llama-3.2-1B o arquivo tem 11,12 bits/peso e volta com o SHA-256 dos dois originais. Ver a [seção 21 dos resultados](docs/05-resultados.md#21-modo-residual-gravado-strata03).
- **Implementado (`strata/0.2`, planos de bits, o primeiro formato):** estratos de sinal, expoente e cada bit da mantissa em ordem estrato-maior; rANS determinístico no expoente; leitura por prefixo; reconstrução no ponto médio; volta bit a bit para `.safetensors`; BF16, FP16, FP32, FP8 e inteiros; leitor que recusa arquivo malformado ou hostil e SHA-256 por stream, por tensor e do original, em dois leitores independentes (JS e Rust). Ver a [seção 18 dos resultados](docs/05-resultados.md#18-segurança-do-leitor).
- **Medido num modelo real:** no Llama-3.2-1B, o `.strata` sem perdas tem 10,59 bits/peso (66% do BF16) e volta com SHA-256 idêntico; o ΔPPL de cada estrato no wikitext-2 está em [`tests/ppl/`](tests/ppl) e a comparação com INT-k na [seção 12 dos resultados](docs/05-resultados.md#12-rans-no-expoente-num-modelo-real-llama-32-1b).
- **Medido, mas ainda não gravado no arquivo:** a genealogia (o filho como diferença do pai).
- **Distribuição sobre o GGUF do Hub:** o Q4_K_M publicado fica intacto como estrato 0, e os níveis e o fecho vão em arquivos ao lado. No Llama-3.1-8B-Instruct (Q4_K_M com imatrix do bartowski) o conjunto tem 11,13 bits/peso, 11,18 GB contra 33,31 GB de BF16 + Q4_K_M + Q5_K_M + Q6_K, e volta aos 4 shards com o SHA-256 do Hugging Face. No Llama-3.1-70B-Instruct (Colab): 98,27 GB contra 291,47 GB (−66%), os 30 shards voltam com o SHA-256 do Hub, mas a volta ao BF16 levou 1,7 h em 8 núcleos. Ver a [seção 31 dos resultados](docs/05-resultados.md#31-distribuição-sobre-o-gguf-do-hub-llama-31-8b-instruct).
- **O GGUF como receita:** com o llama.cpp b3821 e o imatrix publicado, o BF16 do Llama-3.2-1B-Instruct recria as 17 variantes do repositório do bartowski com o SHA-256 do Hub (17,16 GB que passam a ser deriváveis). Só bate com FMA ligada no compilador; o 8B e outras máquinas não foram testados. Ver a [seção 32](docs/05-resultados.md#32-o-gguf-como-receita-o-repositório-inteiro-do-bartowski-sai-do-bf16-llama-32-1b).
- **Os níveis dentro do llama.cpp:** o patch de [`kernel/`](kernel/) põe os níveis como tipos do ggml na CPU e na GPU. No 8B, com bits escolhidos pelo GPTQ, o nível 1 fica a −0,02 bit/peso do k-quant de mesma qualidade no wikitext e a +0,12 no C4; o nível 2 com a tabela linha, a −0,025 e +0,035. Ver [docs/10](docs/10-motor-de-compressao.md) e [docs/11](docs/11-kernel-do-tipo-d.md).
- **Proposto, sem código:** codec por bloco, árvore Merkle por bloco, rANS intercalado para decodificar na GPU e um adaptador que apresenta o `.strata` como `.safetensors`.
- **Próximo passo:** o motor de compressão ([docs/10](docs/10-motor-de-compressao.md)). O nível 1 no C4 é o atraso que sobra; em andamento, o alocador que escolhe o nível de cada grupo de matrizes pelo efeito medido na saída (C2).

## Limites

- A página e as seções 1 a 9 dos resultados usam um modelo sintético e medem o erro nos pesos. Os modelos reais estão da seção 12 em diante: Llama-3.2-1B, Llama-3.1-8B e o 70B (só armazenamento), mais Qwen3-8B no livro de códigos.
- No `strata/0.2`, o rANS é de ordem 0 e só no expoente; os planos da mantissa continuam em deflate. Contexto renderia cerca de 0,07 bit por peso a mais.
- No `strata/0.2`, mesmo com o mesmo codificador de entropia dos dois lados, um INT-k por bloco atinge o mesmo ΔPPL com 0,4 a 0,5 bit por peso a menos que o corte do STRATA, e perto de 8 bits o INT8 por bloco erra menos. Por isso os cortes com perda passaram para o modo residual.
- No modo residual, os níveis ainda não ganham dos k-quants com imatrix no 8B (+0,12 bit/peso no nível 1 fora do domínio da calibração) e precisam do llama.cpp com o patch de `kernel/`. O ganho medido até aqui é de armazenamento, não de qualidade por bit.
- A linha "≈13 GB num `.strata`" da página é uma estimativa e aparece marcada como tal.

---

Pesquisa, formato e protótipo criados por Cantyll com o Claude, em outubro de 2026. Fontes da pesquisa em [docs/01-pesquisa.md](docs/01-pesquisa.md#fontes).
