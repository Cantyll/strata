# Volta ao BF16 decodificando enquanto baixa

Estratos: 0.92 GB · BF16: 2.47 GB · `junta` com tudo no disco: 24 s · 4 núcleos

| rede (medida) | baixar os estratos | baixa e depois junta | **em fluxo** | baixar o BF16 cru |
|---|---:|---:|---:|---:|
| 100 Mbit/s | 74 s | 98 s | **75 s** | 199 s |
| 296 Mbit/s | 25 s | 49 s | **31 s** | 67 s |
| 922 Mbit/s | 8 s | 32 s | **25 s** | 21 s |
| 2049 Mbit/s (sem teto) | 4 s | 28 s | **24 s** | 10 s |
