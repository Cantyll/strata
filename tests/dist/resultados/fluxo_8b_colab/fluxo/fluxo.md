# Volta ao BF16 decodificando enquanto baixa

Estratos: 6.25 GB · BF16: 16.06 GB · `junta` com tudo no disco: 191 s · 12 núcleos

| rede (medida) | baixar os estratos | baixa e depois junta | **em fluxo** | baixar o BF16 cru |
|---|---:|---:|---:|---:|
| 100 Mbit/s | 501 s | 692 s | **518 s** | 1286 s |
| 299 Mbit/s | 167 s | 359 s | **208 s** | 430 s |
| 981 Mbit/s | 51 s | 242 s | **197 s** | 131 s |
| 1232 Mbit/s (sem teto) | 41 s | 232 s | **193 s** | 104 s |
