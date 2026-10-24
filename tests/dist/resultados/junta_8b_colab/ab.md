# junta: antigo × novo no Colab (Llama-3.1-8B)

```
CPU(s):                                  8
Model name:                              Intel(R) Xeon(R) CPU @ 2.20GHz
Thread(s) per core:                      2
sha_ni: False · avx2: True
RAM 54.8 GB · disco livre 220 GB
```

| versão | s | × |
|---|---:|---:|
| antigo --tudo (como no 70B) | 603 | 1.00 |
| novo | 261 | 2.31 |
| novo --tudo | 357 | 1.69 |

70B, que levou 6.128 s com o antigo: estimativa com o novo ≈ 2657 s (proporção do 8B; o 70B não cabe no cache de disco, então pode ficar pior).
