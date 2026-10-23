#!/usr/bin/env bash
# Renderiza uma página do padrão de vídeo em 4 partes paralelas, junta a narração e comprime.
#   ./monta.sh <pagina.html> <narracao.mp3> <duracao_total_s> <saida.mp4>
# A narração entra 0,5 s depois do quadro 0 (OFF da página). Precisa de node + Playwright e ffmpeg.
set -euo pipefail
PAG=$(realpath "$1"); AUD=$2; DUR=$3; OUT=$4
AQUI=$(dirname "$(realpath "$0")")
N=$(python3 -c "print(round($DUR*30))"); Q=$(( (N + 3) / 4 ))
TMP=$(mktemp -d)
for i in 0 1 2 3; do
  a=$((i * Q)); b=$(( (i + 1) * Q )); [ $b -gt $N ] && b=$N
  PAGINA=$PAG node "$AQUI/render.mjs" video $a $b "$TMP/parte$i.mp4" &
done
wait
printf "file '$TMP/parte%d.mp4'\n" 0 1 2 3 > "$TMP/lista.txt"
ffmpeg -y -loglevel error -f concat -safe 0 -i "$TMP/lista.txt" -c copy "$TMP/video.mp4"
ffmpeg -y -loglevel error -i "$TMP/video.mp4" -i "$AUD" \
  -filter_complex "[1:a]adelay=500|500,loudnorm=I=-16:TP=-1.5:LRA=11,apad[a]" -map 0:v -map "[a]" \
  -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -c:a aac -b:a 160k -shortest -movflags +faststart "$OUT"
rm -rf "$TMP"
ffprobe -v error -show_entries format=duration,size -of compact "$OUT"
