#!/bin/bash
# Confere --guarda fp16 no 1B contra o fp32 (c2pt/N1.json, Q4_K_M.json).
cd /home/user/work; S=/home/user/strata
T="--modelo hf1b --texto wiki.test.raw c4.test.raw --chunks 40 --device cpu --fluxo --guarda fp16 --base v16/base"
python3 -u $S/tests/kernel/strata_dkl.py teste $T --niveis c2l/e1.gguf --out v16/N1.json 2>&1 | grep -vi warn
python3 -u $S/tests/kernel/strata_dkl.py teste $T --niveis c2l/e0.gguf --out v16/Q4_K_M.json 2>&1 | grep -vi warn
echo V16_FIM
