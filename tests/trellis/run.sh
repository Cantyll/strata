set -u
W=/home/user/work/kt; cd $W
BIN=$W/ik/build/bin
QS="${QS:-Q8_0 Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S IQ4_XS Q3_K_M IQ4_K IQ4_KS IQ4_KT IQ3_KT IQ2_KT}"
mkdir -p q logs
echo "STAGE quantize"
for t in $QS; do
  [ -f q/$t.gguf ] || { s0=$(date +%s); $BIN/llama-quantize --imatrix bart.imatrix --token-embedding-type q8_0 f16.gguf q/$t.gguf $t 4 > logs/quant_$t.log 2>&1; rc=$?; echo "quant $t rc=$rc secs=$(( $(date +%s)-s0 ))"; }
done
echo "STAGE base"
[ -f base.kld ] || $BIN/llama-perplexity -m f16.gguf -f wiki.test.raw -c 512 --chunks 60 -t 4 --kl-divergence-base base.kld > logs/ppl_f16.log 2>&1
echo "base rc=$? $(grep -o 'Final estimate.*' logs/ppl_f16.log)"
echo "STAGE kl"
for t in $QS; do
  $BIN/llama-perplexity -m q/$t.gguf -c 512 -t 4 --kl-divergence-base base.kld --kl-divergence > logs/kl_$t.log 2>&1
  echo "kl $t rc=$? $(grep -E 'Mean +KLD' logs/kl_$t.log | head -1)"
done
echo "STAGE bench"
for t in $QS; do
  $BIN/llama-bench -m q/$t.gguf -t 4 -p 512 -n 128 -r 3 -o md > logs/bench_$t.log 2>&1
  echo "bench $t rc=$? $(grep -E 'tg128' logs/bench_$t.log | tr -s ' ')"
done
echo ALL_DONE
