GPU: Tesla T4, 40 SMs, L2 4 MB
cópia na GPU: 243 GB/s (leitura + escrita)

| decodificação | bits/peso | µs por matriz 14336×4096 | GB/s lidos | relativo ao uniforme do mesmo k | relativo ao uniforme de 4 bits | confere |
|---|---:|---:|---:|---:|---:|---|
| uniforme, 4 bits | 4.31 | 166.2 | 190 | 1.00× | 1.00× | ok |
| livro (tabela), 4 bits | 4.31 | 173.6 | 182 | 0.96× | 0.96× | ok |
| uniforme, 5 bits | 5.31 | 190.9 | 204 | 1.00× | 0.87× | ok |
| livro (tabela), 5 bits | 5.31 | 199.0 | 196 | 0.96× | 0.84× | ok |
| uniforme, 6 bits | 6.31 | 228.1 | 203 | 1.00× | 0.73× | ok |
| livro (tabela), 6 bits | 6.31 | 291.4 | 159 | 0.78× | 0.57× | ok |

relativo = velocidade (tempo do outro / tempo deste): abaixo de 1 é mais lento.
