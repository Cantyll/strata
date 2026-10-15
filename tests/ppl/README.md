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

Onde rodar: uma GPU com 8 GB ou mais (Colab gratuito com T4 serve) faz a tabela inteira em poucos minutos por configuração. Em CPU funciona, mas cada configuração passa pelo wikitext-2 inteiro (~140 janelas de 2048 tokens) e a tabela completa (13 configurações) leva horas; nesse caso use `--max-chunks 40`, porque a comparação entre configurações continua pareada nos mesmos tokens.

## O que é medido e o que não é

- **Cortados:** todas as `nn.Linear` dentro de `model.layers.*` (q, k, v, o, gate, up, down). Embedding, `lm_head` e normas ficam em BF16 nos dois lados da comparação. `--include-embed` corta também o embedding.
- **Bits/peso do STRATA:** medidos, separando sinal, expoente e os 7 planos da mantissa e comprimindo cada um com deflate-raw, como o core faz. Contam só as matrizes cortadas.
- **Bits/peso do INT-k:** nominais, k bits mais uma escala FP16 a cada 32 pesos (k + 0,5). Não é Q4_K_M nem Q8_0 do llama.cpp: os K-quants usam escalas em dois níveis e escolhem a escala minimizando o erro, então erram menos que este INT-k ingênuo. Comparar com GGUF de verdade exige rodar `llama-perplexity` com os mesmos tokens.
- **RAM:** o corte do STRATA volta a BF16 (16 bits) na memória; não há kernel que calcule direto sobre os planos.
- **Contas em float32** por padrão, para que a diferença venha só dos pesos.
