# ΔPPL real do STRATA

`strata_ppl.py` mede a perplexidade de um modelo no wikitext-2 com os pesos cortados em cada estrato do STRATA (o mesmo `truncMid` de `src/strata-core.js`), contra o BF16 original e contra INT4/5/6/8 simétrico em bloco de 32 (o mesmo esquema de `blockQuantErr`).

A saída é uma tabela de bits por peso no disco contra ΔPPL, em `tests/ppl/resultados/<modelo>.md` e `.json`.

## Rodar

Precisa de Python 3.10+, acesso ao Hugging Face e `node` (só para a conferência de paridade com o core JS).

```
pip install torch transformers datasets safetensors numpy
huggingface-cli login     # aceite antes a licença em https://huggingface.co/meta-llama/Llama-3.2-1B
python tests/ppl/strata_ppl.py --model meta-llama/Llama-3.2-1B
```

Rodada curta, para conferir que funciona antes da longa:

```
python tests/ppl/strata_ppl.py --model meta-llama/Llama-3.2-1B --max-chunks 4 --depths 2,7 --int-bits 4
```

Colab: abra [`strata_ppl_colab.ipynb`](https://colab.research.google.com/github/Cantyll/strata/blob/claude/project-thread-2p2spc/tests/ppl/strata_ppl_colab.ipynb), escolha GPU T4 e execute tudo. Ele usa `unsloth/Llama-3.2-1B`, cópia aberta do mesmo modelo, sem login.

Onde rodar: uma GPU com 8 GB ou mais (Colab gratuito com T4 serve) faz a tabela inteira em poucos minutos por configuração. Em CPU funciona, mas cada configuração passa pelo wikitext-2 inteiro (~140 janelas de 2048 tokens) e a tabela completa (13 configurações) leva horas; nesse caso use `--max-chunks 40`, porque a comparação entre configurações continua pareada nos mesmos tokens.

## O que é medido e o que não é

- **Cortados:** todas as `nn.Linear` dentro de `model.layers.*` (q, k, v, o, gate, up, down). Embedding, `lm_head` e normas ficam em BF16 nos dois lados da comparação. `--include-embed` corta também o embedding.
- **Bits/peso do STRATA:** medidos, separando sinal, expoente e os 7 planos da mantissa e comprimindo cada um como o core v0.2 faz: rANS de ordem 0 no expoente, deflate-raw no resto. Contam só as matrizes cortadas. A tabela de `resultados/` foi gerada antes do rANS, com deflate em tudo; os números com rANS estão em [`docs/05-resultados.md`](../../docs/05-resultados.md).
- **Bits/peso do INT-k:** nominais, k bits mais uma escala FP16 a cada 32 pesos (k + 0,5). Não é Q4_K_M nem Q8_0 do llama.cpp: os K-quants usam escalas em dois níveis e escolhem a escala minimizando o erro, então erram menos que este INT-k ingênuo. Para os GGUFs de verdade, use `--gguf` (abaixo).
- **RAM:** o corte do STRATA volta a BF16 (16 bits) na memória; não há kernel que calcule direto sobre os planos.
- **Contas em float32** por padrão, para que a diferença venha só dos pesos.

## GGUFs do llama.cpp

`--gguf Q4_K_M=modelo-Q4_K_M.gguf,Q5_K_M=modelo-Q5_K_M.gguf` mede GGUFs na mesma tabela: o `gguf-py` (pasta `gguf-py` do llama.cpp, `pip install -e`) desquantiza as mesmas matrizes e desfaz a permutação de q/k do `convert_hf_to_gguf`. Bits/peso = bytes reais dessas matrizes no GGUF. `--tag` muda o nome dos arquivos de saída, e o JSON é regravado a cada linha.

[`strata_vs_kquants_colab.ipynb`](https://colab.research.google.com/github/Cantyll/strata/blob/claude/project-thread-qm128x/tests/ppl/strata_vs_kquants_colab.ipynb) compila o `llama-quantize`, gera Q3_K_M a Q8_0 e roda estratos, INT8 progressivo e GGUFs juntos numa T4. Resultado na CPU: [`docs/05-resultados.md`, seção 13](../../docs/05-resultados.md#13-contra-os-k-quants-do-llamacpp-q4_k_m-q5_k_m-q6_k).
