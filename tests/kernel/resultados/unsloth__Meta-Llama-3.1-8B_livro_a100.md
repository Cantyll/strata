# STRATA, livro de códigos aninhado (Llama-3.1-8B, caminho F16)

GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, 8.0, 580.82.07. KL contra o BF16 nos 100 primeiros trechos do wikitext-2 raw de teste; imatrix com 125 trechos do texto de treino. Bits por peso nas matrizes das camadas.

| modelo | bits/peso | KL (×1000) | mesmo 1º token | erro ponderado |
|---|---:|---:|---:|---:|
| F16_Q3_K_M_imat | 3.844 | 76.62 ± 1.20 | 87.66% | 1.6784% |
| bloco_N0 | 4.250 | 33.52 ± 0.58 | 91.66% | 0.4386% |
| F16_IQ4_XS_imat | 4.274 | 31.66 ± 0.53 | 91.80% | 0.4666% |
| F16_Q4_K_S_imat | 4.536 | 28.96 ± 0.51 | 92.03% | 0.3949% |
| F16_Q4_K_M_imat | 4.797 | 25.32 ± 0.49 | 92.68% | 0.3850% |
| bloco_N1 | 5.250 | 12.38 ± 0.21 | 94.78% | 0.1472% |
| bloco_ind_N1 | 5.250 | 9.32 ± 0.19 | 95.34% | 0.1110% |
| F16_Q5_K_S_imat | 5.500 | 8.21 ± 0.16 | 95.74% | 0.1012% |
| F16_Q5_K_M_imat | 5.653 | 7.55 ± 0.14 | 96.01% | 0.0991% |
| bloco_N2 | 6.250 | 4.28 ± 0.07 | 96.86% | 0.0391% |
| bloco_ind_N2 | 6.250 | 3.61 ± 0.06 | 97.07% | 0.0320% |
| F16_Q6_K_imat | 6.562 | 3.31 ± 0.06 | 97.12% | 0.0256% |
| F16_Q6_K_Q8mix_imat | 7.457 | 1.57 ± 0.02 | 98.00% | 0.0166% |
| F16_Q8_0 | 8.500 | 0.57 ± 0.01 | 98.96% | 0.0028% |

## Régua (curva = k-quants com imatrix em F16)

| candidato | bits/peso | KL ×1000 | curva no mesmo tamanho | razão | bits a mais | bits a mais (curva com IQ4_XS) |
|---|---:|---:|---:|---:|---:|---:|
| bloco_N0 | 4.250 | 33.52 | 43.29 | 0.77 | -0.182 | +0.004 |
| bloco_N1 | 5.250 | 12.38 | 12.25 | 1.01 | +0.006 | +0.006 |
| bloco_N2 | 6.250 | 4.28 | 4.39 | 0.98 | -0.027 | -0.027 |
| bloco_ind_N1 | 5.250 | 9.32 | 12.25 | 0.76 | -0.171 | -0.171 |
| bloco_ind_N2 | 6.250 | 3.61 | 4.39 | 0.82 | -0.216 | -0.216 |

## Veredito pela régua de docs/09

- **bloco**: sucesso (níveis 0/1/2: +0.004 / +0.006 / -0.027; soma -0.017)
- **bloco_ind**: ganho forte (níveis 0/1/2: +0.004 / -0.171 / -0.216; soma -0.383)

Referência (seção 30, tipos D sobre o Q4_K_M do Hub): 0 / +0,23 / +0,11 bit/peso.

**Dominância** (menos bits e menos KL ao mesmo tempo, sem depender da interpolação):

- bloco_N0: nenhum
- bloco_N1: nenhum
- bloco_N2: nenhum
- bloco_ind_N1: nenhum
- bloco_ind_N2: nenhum

## Velocidade na GPU, llama.cpp sem patch (tok/s)

| modelo | geração (tg128) | prompt (pp512) |
|---|---:|---:|
| Q4_K_M_imat | 154.9 | 4617 |
| IQ4_XS_imat | 195.2 | 4704 |
| Q5_K_M_imat | 145.0 | 4457 |
| Q6_K_imat | 117.9 | 4278 |
| IQ4_XS / Q4_K_M | 1.26 | 1.02 |
