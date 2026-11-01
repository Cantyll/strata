#!/bin/bash
# C1 + C2 no Llama-3.2-3B, no contêiner (regra em docs/10, "O 3B no contêiner"). Retomável: cada passo pula o que já saiu.
set -u
W=/home/user/work; S=/home/user/strata; B=$W/llama.cpp/build/bin; G=$W/b3; R=$W/b3/res; HF=$W/hf3b
TAM="python3 $S/tests/kernel/resultados/c2_3b/tam_tensores.py"
mkdir -p $G $R; cd $W
while pgrep -f "^/bin/bash ./v16.sh" >/dev/null || pgrep -f "^/bin/bash ./dl3b.sh" >/dev/null; do sleep 30; done
# o 1B já está no repositório; os arquivos grandes dele saem para caber o 3B (refazíveis)
rm -f c2l/e0.gguf c2l/e1.gguf c2l/e2.gguf c2pt/base.* g/BF16.gguf hf1b/model.safetensors g2/Q4_K_M.gguf
baixa() {
  [ -s $HF/model-00001-of-00002.safetensors ] && [ -s $HF/model-00002-of-00002.safetensors ] && return
  python3 -c "from huggingface_hub import snapshot_download as s; s('unsloth/Llama-3.2-3B', allow_patterns=['*.json','*.safetensors','tokenizer*'], local_dir='$HF')" > $G/baixa.log 2>&1
}
T="--modelo $HF --texto wiki.test.raw c4.test.raw --chunks 40 --device cpu --fluxo --guarda fp16 --base $R/base"
te() {  # te nome args...
  [ -s $R/$1.json ] && return
  python3 -u $S/tests/kernel/strata_dkl.py teste $T --out $R/$1.json "${@:2}" > $R/$1.log 2>&1
  grep -E "KL ×1000|rror|Trace" $R/$1.log | sed "s/^/$1 /"
  [ -s $R/$1.json ] || { echo "falhou teste $1"; exit 1; }
}
echo "início $(date +%H:%M)"
# 1. BF16, imatrix (125 trechos da calibração misturada, do BF16, como no 1B) e Q4_K_M (nível 0)
if [ ! -s $G/Q4_K_M.gguf ] || [ ! -s $R/N2.json ]; then
  baixa
  [ -s $G/BF16.gguf ] || python3 $W/llama.cpp/convert_hf_to_gguf.py $HF --outtype bf16 --outfile $G/BF16.gguf > $G/conv.log 2>&1 || { echo "falhou conv"; exit 1; }
fi
[ -s $G/imatrix.gguf ] || $B/llama-imatrix -m $G/BF16.gguf -f $W/calib_mix.raw -c 512 --chunks 125 -t 4 -o $G/imatrix.gguf > $G/imatrix.log 2>&1
[ -s $G/Q4_K_M.gguf ] || $B/llama-quantize --imatrix $G/imatrix.gguf $G/BF16.gguf $G/Q4_K_M.gguf Q4_K_M > $G/Q4_K_M.log 2>&1
[ -s $R/tam_Q4_K_M.txt ] || $TAM $G/Q4_K_M.gguf > $R/tam_Q4_K_M.txt
echo "BF16/imatrix/Q4_K_M ok $(date +%H:%M)"
# 2. frente dos k-quants, lidos direto do GGUF quantizado; cada um sai depois de medido
te Q4_K_M --niveis $G/Q4_K_M.gguf
for q in Q5_K_S Q5_K_M Q6_K Q8_0; do
  [ -s $R/$q.json ] && continue
  im="--imatrix $G/imatrix.gguf"; [ $q = Q8_0 ] && im=""
  $B/llama-quantize $im $G/BF16.gguf $G/q.gguf $q > $G/q_$q.log 2>&1
  $TAM $G/q.gguf > $R/tam_$q.txt
  te $q --niveis $G/q.gguf
  rm -f $G/q.gguf
done
echo "frente ok $(date +%H:%M)"
# 3. Hessianas (500 trechos do calib_big, uma passada, bf16, só o triângulo) e níveis P,DL com GPTQ
if [ ! -f $G/niveis.ok ]; then
  if [ ! -s $G/hess/info.json ] || [ ! -s $G/hess/camada_0.npz ]; then   # com --apaga-hess, um GPTQ interrompido deixa a pasta pela metade
    rm -rf $G/hess $G/e1.gguf $G/e2.gguf
    baixa
    python3 -u $S/tests/kernel/strata_hessiana.py --modelo $HF --texto $W/calib_big.raw --chunks 500 --out $G/hess --fluxo --dtype bf16 --triangulo > $G/hess.log 2>&1
    [ -s $G/hess/info.json ] || { echo "falhou hess"; exit 1; }
  fi
  tail -1 $G/hess.log
  rm -f $HF/*.safetensors          # disco: volta a baixar depois do GPTQ
  python3 -u $S/tests/kernel/strata_gptq_niveis.py --bf16 $G/BF16.gguf --gguf $G/Q4_K_M.gguf --imatrix $G/imatrix.gguf \
    --hess $G/hess --apaga-hess --amortece 0.1 --pontos P,DL --escolha gptq --out $G/e1.gguf $G/e2.gguf --relatorio $R/rel_gptq.json > $G/gptq.log 2>&1 \
    && touch $G/niveis.ok
  tail -1 $G/gptq.log
  [ -f $G/niveis.ok ] || { rm -rf $G/hess; echo "falhou gptq"; exit 1; }
  rm -rf $G/hess
fi
echo "níveis ok $(date +%H:%M)"
N="--niveis $G/Q4_K_M.gguf $G/e1.gguf $G/e2.gguf"
[ -s $R/erros.json ] || python3 -u $S/tests/demanda/strata_mistura.py erros --bf16 $G/BF16.gguf --imatrix $G/imatrix.gguf $N --out $R/erros.json > $R/erros.log 2>&1 || { echo "falhou erros"; exit 1; }
rm -f $G/BF16.gguf
baixa
# 4. níveis inteiros
te N1 --niveis $G/e1.gguf
te N2 --niveis $G/e2.gguf
echo "níveis inteiros ok $(date +%H:%M)"
# 5. ΔKL por grupo (textos da medida, 3 trechos de cada, faixas de 4 camadas, partindo do nível 1)
M="--modelo $HF $N --texto wiki.valid.raw c4.medida.raw --chunks 3 --device cpu --fluxo --guarda fp16"
[ -s $R/dkl.json ] || python3 -u $S/tests/kernel/strata_dkl.py mede $M --banda 4 --ref 1 --out $R/dkl.json > $R/dkl.log 2>&1 || { echo "falhou mede"; exit 1; }
echo "mede ok $(date +%H:%M)"
# 6. orçamentos: Q4_K_M + 0,35 / 0,55 / 0,75 do caminho entre o nível 0 e o nível 2 (custos do plano, tabela de 16 bytes)
ORC=$(python3 - $G/Q4_K_M.gguf $R/dkl.json <<'PY'
import json, os, sys
d = json.load(open(sys.argv[2])); q4 = os.path.getsize(sys.argv[1])
nivel = sum(n // 256 * 32 + 16 for n in d["pesos"].values())
print(" ".join(f"{(q4 + f * 2 * nivel) / 2**30:.4f}" for f in (0.35, 0.55, 0.75)))
PY
)
echo "orçamentos (GiB): $ORC"
for o in $ORC; do
  [ -s $R/p_dkl_$o.json ] || python3 $S/tests/kernel/strata_dkl.py plano --dkl $R/dkl.json --q4km $G/Q4_K_M.gguf --gib $o --tabela 16 --out $R/p_dkl_$o.json
  [ -s $R/p_mse_$o.json ] || python3 $S/tests/demanda/strata_mistura.py grava --erros $R/erros.json --q4km $G/Q4_K_M.gguf --gib $o --tabela 16 $N --plano-out $R/p_mse_$o.json > $R/g_mse_$o.log 2>&1
  for a in dkl mse; do te ${a}_$o $N --plano $R/p_${a}_$o.json; done
done
echo "planos ok $(date +%H:%M)"
# 7. a soma prevê a medida?
PL=$(for a in dkl mse; do for o in $ORC; do echo -n "$R/p_${a}_$o.json "; done; done)
[ -s $R/avalia.json ] || python3 -u $S/tests/kernel/strata_dkl.py avalia $M --plano $PL --out $R/avalia.json > $R/avalia.log 2>&1
grep "KL medida" $R/avalia.log
echo B3_FIM $(date +%H:%M)
