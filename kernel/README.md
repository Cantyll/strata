# Kernel do STRATA: tipos Q4_K_R<k>, Q6_K_R<k> e blocos montados na carga

Referência escalar em C99 dos tipos que rodam os níveis do modo residual (especificação B.8). Ela define o resultado;
as versões AVX2, NEON e CUDA que vierem depois precisam reproduzi-lo bit a bit.

- `strata_q4kr.h`: layout do bloco e fórmula.
- `strata_q4kr.c`: desquantização de uma linha, quantização das ativações em Q8_K e produto escalar inteiro.
- `strata_q4kr_avx2.c`: produto escalar em AVX2, com o mesmo float da referência, bit a bit (tipos R), e o dos blocos
  montados na carga (tipos P, que somam em float vetorial como o ggml).
- Velocidade contra o ggml: `tests/kernel/bench_vec_dot.c` (instruções no topo do arquivo).
- `ggml/`: patch que põe os blocos montados como tipos do llama.cpp (ver "Dentro do llama.cpp" no fim).
- Teste num modelo real: `python tests/kernel/test_q4kr.py --model unsloth/Llama-3.2-1B --gguf Q4_K_M.gguf`.

## O bloco

| bytes | conteúdo |
|---|---|
| 0–143 | o bloco Q4_K do estrato 0, byte a byte |
| 144–175 | plano do nível 1: 1 bit por peso |
| 176–207 | plano do nível 2 |
| 208–239 | plano do nível 3 |

Q4_K_R1 tem 176 bytes por 256 pesos (5,5 bits/peso, o tamanho do Q5_K), Q4_K_R2 208 (6,5, o do Q6_K), Q4_K_R3 240 (7,5).
Subir um nível é acrescentar 32 bytes no fim de cada bloco; o que já está na memória não muda.

## A conta

Com `a = d·sc` e `m = dmin·mn` do sub-bloco, como no Q4_K, `r` = os k bits dos planos (nível 1 mais significativo):

    t = 2^(k+1)·q4 + 2r + 1 − 2^k        v = a · t / 2^(k+1) − m

`v` é o centro da sub-célula. O mínimo é o do Q4_K, sem requantizar (a perda do GGUF Q5_K vinha de requantizá-lo).
`t` cabe em int8 até k = 2 (−3…123), então o produto escalar contra ativações Q8_K é uma soma de inteiros por sub-bloco:

    dot = Σ_blocos d_x · ( d/2^(k+1) · Σ_j sc_j · Σ t·q8  −  dmin · Σ_j mn_j · Σ q8 )

## Q6_K_R<k>

Para as matrizes que o Q4_K_M guarda em Q6_K. Bloco = o Q6_K byte a byte (210 bytes: `ql[128]`, `qh[64]`, 16 escalas
int8, `d`) seguido dos mesmos k planos de 32 bytes: 242, 274 e 306 bytes (7,56, 8,56 e 9,56 bits/peso nessas matrizes).
Com `a = d·sc_j` (sub-bloco de 16 pesos, escala com sinal) e `q` de −32 a 31: `t = 2^(k+1)·q + 2r + 1 − 2^k`, `v = a·t/2^(k+1)`.
`r` conta no sentido de `q` crescente (valor decrescente quando `sc < 0`, metade dos sub-blocos). A soma inteira do bloco
vai em 64 bits: com k = 3 o pior caso passa de 2^31.

## Estado

- Conferido no Llama-3.2-1B, as 112 matrizes (96 Q4_K com 830 M pesos e 16 Q6_K com 143 M), k = 1, 2, 3: desquantização
  igual bit a bit à fórmula em float32, igual à emulação que deu os ΔPPL (seções 20 e 22 dos resultados; diferença
  abaixo de 1,3e-6 do passo, arredondamento float32 contra float64), produto escalar com erro relativo ~1e-8.
- Os blocos `Q4_K_R` do leitor em Rust (`blocos_q4kr`, arquivo 0.3) são iguais byte a byte aos que o teste monta.
- AVX2 dos tipos R: o produto escalar dá o mesmo float da referência (conferido em `test_q4kr.py`). Os números de
  velocidade estão na tabela da seção seguinte. Duas ideias medidas em protótipo e deixadas de fora: planos na ordem
  do `qh` do Q5_K (bit do peso 32j + l no byte l, bit j) não muda o nível 1 e ganha 4 a 11% nos níveis 2 e 3, pouco
  para mudar o formato; somar em vetor de float entre blocos ganha uns 5% e perderia a igualdade bit a bit (adotado
  só nos blocos montados, abaixo).
- Os blocos montados já são tipos do ggml (seção "Dentro do llama.cpp"). Falta: NEON e CUDA, e medir num modelo maior.

## Blocos montados na carga (Q4_K_P<k>, Q6_K_P<k>)

O arquivo continua em camadas (estrato 0 + planos), que é o que permite baixar e guardar um arquivo só. Mas na RAM
não há por que manter os bits separados: ao carregar o modelo, cada bloco R vira, uma vez, um bloco em que o índice
inteiro de cada peso já está junto, como nos k-quants. Os valores são os mesmos, bit a bit (a desquantização e o
produto escalar escalar dos blocos montados são iguais aos do tipo R); só muda o arranjo. É como receber um móvel
em peças numeradas: no transporte as peças vão separadas, mas você monta uma vez ao chegar e depois só usa.

| tipo | bytes / 256 pesos | arranjo |
|---|---:|---|
| Q4_K_P1 | 176 (5,5 b) | idêntico ao `block_q5_K` do ggml: d, dmin, escalas/mínimos do Q4_K, `qh[32]`, `qs[128]`; u = 2·q4 + r (5 bits) |
| Q4_K_P2 | 208 (6,5 b) | d, dmin, escalas/mínimos, `qs[128]` (4 bits baixos de u), `qh[64]` (2 bits altos); u = 4·q4 + r (6 bits) |
| Q6_K_P1 | 242 (7,56 b) | o próprio bloco Q6_K_R1: `ql[128]`, `qh[64]`, 16 escalas int8, d, plano de 32 bytes |
| Q6_K_P2 | 274 (8,56 b) | w = 4·q + r em int8 (ordem do Q6_K), 16 escalas int8, d |

`t = 2u + 1 − 2^k` (ou `2w + 1 − 2^k`) é o mesmo `t` do tipo R, então o produto escalar é um `maddubs` por 32 pesos, como
no Q5_K/Q6_K, mais um termo das bsums por bloco. O mínimo continua exato (é o do Q4_K, não requantizado). O Q6_K_P2
ocupa o mesmo que o Q6_K_R2. O Q6_K_P1 é o bloco do Q6_K_R1 sem mudar nada: o produto escalar junta 2u + r em 7 bits
sem sinal e usa um `maddubs` só, então não precisa de 1 bit/peso a mais na RAM (antes, com w em int8, gastava 274
bytes). A montagem custa ~3 a 4 ns por peso numa thread (~4 s para 1 bilhão de pesos, uma vez por carga, paralelizável).

A versão AVX2 dos blocos montados soma as somas inteiras de cada bloco num vetor de 8 floats, como o ggml, e por
isso pode diferir da referência escalar nos últimos bits (erro relativo medido ~1e-8; o teste aceita até 1e-6).

Velocidade do produto escalar (matriz × vetor, ns por peso; blocos aleatórios; CPU de nuvem compartilhada, o menor de
6 tempos, ruidoso e só indicativo). Uma thread, matriz 4096 × 4096 (limitado pela conta):

| ggml | | STRATA no arquivo (R) | | STRATA montado (P) | |
|---|---:|---|---:|---|---:|
| Q5_K | 0,042 | Q4_K_R1 | 0,054 | Q4_K_P1 | 0,043 |
| Q6_K | 0,034 | Q4_K_R2 | 0,069 | Q4_K_P2 | 0,044 |
| Q6_K | 0,034 | Q6_K_R1 | 0,058 | Q6_K_P1 | 0,037 |
| Q6_K | 0,034 | Q6_K_R2 | 0,081 | Q6_K_P2 | 0,038 |

4 threads, matriz 16384 × 8192 (mais perto do caso real, limitado pela memória): Q5_K 0,012, Q6_K 0,016;
Q4_K_P1 0,013, Q4_K_P2 0,015, Q6_K_P1 0,022, Q6_K_P2 0,024; Q4_K_R1 0,017, Q4_K_R2 0,023.

No modelo inteiro (85% dos pesos em matrizes Q4_K, 15% em Q6_K), contra o k-quant de mesma qualidade:

| | antes (R) | montado (P), 1 thread | montado (P), 4 threads |
|---|---:|---:|---:|
| nível 1 contra Q5_K_M | ~1,35× | 1,03× (Q6_K_P1) a 1,11× (Q6_K_R1) | 1,13× a 1,24× |
| nível 2 contra Q6_K | ~2× | 1,27× | 1,02× |

O nível 1 empata com o Q5_K na matriz Q4_K. O que sobra no nível 2 com uma thread é o desembrulho das escalas de 6
bits do Q4_K e o termo do mínimo, que o Q6_K não tem: com escalas guardadas já desembrulhadas (16 bytes em vez de 12,
+0,125 bit/peso) o Q4_K_P cai para 0,039 em protótipo, mais rápido que o Q5_K, mas gasta RAM; ficou de fora porque com
várias threads o gargalo é a memória, e ali o nível 2 já empata.

## Base do GGUF e planos separados (tipos S, precisão sob demanda)

`strata_separado.h` / `strata_separado_avx2.c`: o mesmo produto escalar dos tipos R, mas com o bloco Q4_K/Q6_K num buffer
(que pode ser o GGUF do Hub mapeado do disco) e o plano de cada nível num buffer próprio de 32 bytes por bloco. Na RAM os
planos ficam na ordem do `qh` do Q5_K (byte l, bit j = peso 32j + l), e `strata_plano_para_qh` converte do arranjo do
arquivo ao subir de nível (0,04 ns por peso). O resultado é igual bit a bit ao dos tipos P. Medido no E2 do
[docs/08](../docs/08-precisao-sob-demanda.md#6-resultado-do-e2-microbenchmark-container): empata ou vence o P no Q4_K
(níveis 1 e 2) e no Q6_K nível 1; no Q6_K nível 2 fica em 0,75 do P, que por isso continua sendo montado.
No ggml (patch de `ggml/`) são os tipos `STRATA_Q4_K_S1` (51), `STRATA_Q4_K_S2` (52) e `STRATA_Q6_K_S1` (53), só na
CPU com AVX2 e só na memória: os ponteiros dos planos ficam no `op_params` do tensor (`ggml_strata_s_planos`), e o
`llama_strata_tensor` dá acesso ao tensor para trocar o nível entre dois tokens (E3, `tests/demanda/strata_subir.cpp`).

## Dentro do llama.cpp

`ggml/strata-tipos.patch` acrescenta ao ggml (commit `2a66703` do llama.cpp) os tipos `STRATA_Q4_K_P1` (43),
`STRATA_Q4_K_P2` (44), `STRATA_Q6_K_P1` (45) e `STRATA_Q6_K_P2` (46): tamanho do bloco, desquantização (a referência
escalar) e produto escalar contra Q8_K (a versão AVX2), usado na geração, uma linha de entrada por vez. Na leitura do
prompt (8 linhas ou mais de uma vez) os tipos entram no caminho *tiled* do ggml-cpu, o mesmo do Q5_K e do Q6_K: cada
bloco montado é desembrulhado em códigos sem sinal que o tile já sabe multiplicar, com a conta
`d·sc·(q − BIAS) − dmin·mn` do tile:

| tipo | código q | BIAS | d do tile | dmin do tile, mínimo |
|---|---|---:|---|---|
| Q4_K_P<k> | 2u + 1 | 2^k | d / 2^(k+1) | dmin, mínimos do Q4_K |
| Q6_K_P<k> | w + 128 | 128 | d / 2^k | d·(2^k − 1)/2^(k+1), as próprias escalas |

O tile soma em inteiros (VNNI com AVX-512; com só AVX2, o truque do sinal do ggml, `maddubs(|q − BIAS|, ±ativação)`,
que cabe em int16 porque |q − BIAS| ≤ 128). Conferido contra o produto escalar com pesos e ativações reais do
Llama-3.2-1B, nas duas versões: diferença máxima de 4e-7 da saída. `GGML_CPU_TILED_MM=0` desliga o caminho, para comparar.

    sh kernel/ggml/aplicar.sh PASTA_DO_LLAMA_CPP        # patch + cópia dos arquivos do kernel
    cmake -B build -DGGML_NATIVE=ON && cmake --build build -j --target llama-bench llama-perplexity llama-cli
    python tests/kernel/strata_gguf_niveis.py --gguf Q4_K_M.gguf --nivel 1 --out nivel1.gguf

`strata_gguf_niveis.py` faz no conversor o que o leitor faria na carga: parte do Q4_K_M e dos pesos BF16, monta os
blocos R (os mesmos do `.strata` 0.3) e grava os blocos montados num GGUF. As 112 matrizes das camadas ocupam 5,80
bits/peso no nível 1 e 6,80 no nível 2 (Q5_K_M 5,66, Q6_K 6,56).

Conferência no container (Llama-3.2-1B, 8 trechos de 512 tokens do wikitext-2, `llama-perplexity`), divergência KL contra
o BF16, que é bem menos ruidosa que a perplexidade numa amostra pequena:

| modelo | KL (×1000) | mesmo 1º token |
|---|---:|---:|
| Q4_K_M | 42,8 ± 1,7 | 90,0% |
| Q5_K_M | 15,1 ± 0,6 | 93,8% |
| STRATA nível 1 | 14,5 ± 0,5 | 93,5% |
| Q6_K | 7,4 ± 0,3 | 95,8% |
| STRATA nível 2 | 5,3 ± 0,2 | 95,8% |

A ordem é a da emulação (nível 1 ≈ Q5_K_M, nível 2 melhor que Q6_K): os tipos rodam certo dentro do llama.cpp.
Tokens por segundo no container variaram demais entre rodadas para valer como medida; a medição de verdade é o
notebook `tests/kernel/strata_llamacpp_colab.ipynb`, que roda tudo numa máquina do Colab alternando os modelos.
Primeira rodada (EPYC de 22 núcleos, seção 23 dos resultados): qualidade confirmada no texto inteiro, geração a 0,97–1,00
do k-quant de mesma qualidade, leitura do prompt a 0,30–0,33 porque os tipos ainda não passavam pelo caminho tiled
(corrigido depois). Segunda rodada, com o caminho tiled (Xeon de 6 núcleos): leitura do prompt a 0,94 (nível 1) e 1,01
(nível 2), geração a 0,97–0,98 (nível 1) e 0,86–0,90 (nível 2).

## Sub-células por tensor (Q4_K_D<k>, Q6_K_D<k>)

A vertente 3 da seção 28 de `docs/05-resultados.md`: os mesmos blocos dos tipos P, mas cada filho de uma célula fica
numa posição lida de uma tabela de 16 bytes do tensor (`strata_q4kr.h`, especificação B.8), em vez do centro fixo. O
nível 0 continua o Q4_K_M byte a byte. A tabela depende só da classe do código do nível 0 (menor, maior ou meio) e dos
bits dos níveis: a variante "borda" de `tests/kernel/strata_escala_nivel.py --por borda`. No Llama-3.2-1B ela dá a
mesma KL que um deslocamento por código (16 ou 64 por tensor) e cabe num registrador, o que a tabela por código não faz
no Q6_K (64 códigos).

- **CPU** (`strata_q4kr_avx2.c`, `q4kd_dot` / `q6kd_dot`): por 32 pesos, dois `maddubs`, um com o código do nível 0 e
  outro com `|δ|` contra as ativações com o sinal de `δ`; `δ` sai de um `pshufb` na tabela, a classe de dois `cmpeq`. A
  soma do bloco é `d·(Σsc·c·a + Σsc·δ·a/2^(k+4)) − dmin·Σmn·a`. No prompt, o caminho tiled do ggml-cpu faz duas
  passadas no mesmo acumulador: uma com os códigos c (como um Q4_K ou Q6_K), outra com δ + 128 e o d do tile dividido
  por 2^(k+4). A tabela chega por uma variável local da thread, posta antes de cada pedaço da multiplicação.
- **CUDA** (`cuda/strata-vecdot.cuh`, `cuda/strata-dequant.cuh`): no MMVQ, dois `dp4a` e a classe por `__vcmpeq4`; `δ`
  de quatro pesos sai de um `__byte_perm` na tabela. A tabela chega ao kernel num campo novo de
  `ggml_cuda_mm_fusion_args_device` (a fusão fica desligada para os tipos D); na desquantização para o cuBLAS, por uma
  variável do host com o intervalo de memória do tensor, conferido a cada lançamento. `MUL_MAT_ID` (MoE) ainda não.
- **llama.cpp**: o carregador lê a chave `strata.tab.<tensor>` do GGUF e a copia para os `op_params` do tensor; as
  cópias de tensor do agendador levam os `op_params` junto. Um tensor D sem a marca (`tab[15] = 0x53`) aborta com o nome.
- **Conferências**: `tests/kernel/test_q4kd.c` (AVX2 = referência = desquantização, e a tabela padrão dá os tipos P),
  `tests/kernel/strata_cuda_host.sh` (os kernels CUDA na CPU, os 8 tipos) e `tests/kernel/strata_cuda_check.c` (GPU
  contra CPU, agora com os tipos D e uma tabela qualquer). `tests/kernel/strata_gguf_niveis.py --sub-celulas --imatrix`
  grava os GGUFs; `--conferir` compara a desquantização do C com a conta em float64 em cada tensor.

## Na GPU (CUDA)

O mesmo patch põe os quatro tipos no backend CUDA do llama.cpp; os kernels ficam em `cuda/` e o `aplicar.sh` os copia
para `ggml/src/ggml-cuda`. Compila com `-DGGML_CUDA=ON`. A GPU usa dois caminhos, como para os k-quants que o MMQ não
cobre:

- **geração e lotes de até 8 tokens** (MMVQ): `strata-vecdot.cuh`, produto escalar de um pedaço do bloco montado contra
  ativações em Q8_1 com `dp4a`. O Q4_K_P segue o `vec_dot_q5_K_q8_1` (o P1 tem o layout do Q5_K; o P2 tira 2 bits
  altos por peso em vez de 1) e devolve `d/2^(k+1)·(2·Σsc·u·a + (1 − 2^k)·Σsc·a) − dmin·Σmn·a`; o Q6_K_P2 soma `w·a`
  direto (os pesos já estão em int8, na ordem natural) e devolve `d·sc/2^(k+1)·(2·Σw·a + (1 − 2^k)·Σa)`; o Q6_K_P1 monta
  2u + r (= w + 64) de quatro pesos a partir de `ql`, `qh` e do plano e desconta o 64 por Σa. Os blocos Q6_K_P (242 e
  274 bytes) só são alinhados a 2 bytes: os inteiros dos pesos são lidos de 16 em 16 bits (`get_int_b2`);
- **leitura do prompt** (lotes maiores): `strata-dequant.cuh` desquantiza a matriz para FP16 e o cuBLAS multiplica. É o
  caminho mais lento dos dois; o MMQ (multiplicação direto dos blocos, que os k-quants usam) ainda não aceita os tipos.

Conferências no container, sem GPU: os kernels compilam com o `nvcc` (CUDA 13, arquitetura 7.5), e
`tests/kernel/strata_cuda_host.sh` roda a conta dos dois kernels na CPU (os mesmos `.cuh`, com um shim de
`tests/kernel/cuda_shim`) contra a desquantização de referência em C e um produto escalar em float64: erro 0 na
desquantização e ~1e-9 no produto escalar, em blocos aleatórios e reais do Llama-3.2-1B, e uma mutação no kernel é
detectada. A conferência numa GPU de verdade é `tests/kernel/strata_cuda_check.c`, que multiplica matrizes dos quatro
tipos (e Q5_K/Q6_K de controle) nos backends CPU e CUDA, com lotes de 1 a 512, contra uma referência em float64; o
notebook `tests/kernel/strata_8b_gpu_colab.ipynb` compila e roda esse teste antes de medir o Llama-3.1-8B na GPU.
