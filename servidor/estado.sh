#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  bash servidor/estado.sh
set -euo pipefail
ssh fsotoj@172.28.230.10 'bash -s' <<'REMOTE'
P=/home/fsotoj/sam2_stk
echo "== contenedor =="
docker ps -a --filter name=sam2stk --format '  {{.Names}}  {{.Status}}  (exit {{.State}})' 2>/dev/null || echo "  (ninguno)"

PL="$(ls -t $P/logs/par_*_g*.log 2>/dev/null)"
if [ -n "$PL" ]; then
  TSX="$(ls -t $P/logs/par_*_g*.log | head -1 | sed -E 's/.*par_([0-9_]+)_g[0-9]+\.log/\1/')"
  echo; echo "== BARRIDO (lote $TSX) =="
  VIVOS=0
  for f in $P/logs/par_${TSX}_g*.log; do
    g="$(echo $f | sed -E 's/.*_g([0-9]+)\.log/\1/')"
    n="$(grep -c '^>>> ' "$f" 2>/dev/null | head -1)"
    t="$(grep -m1 -oE '\-> [0-9]+ combinaciones' "$f" 2>/dev/null | grep -oE '[0-9]+' | head -1)"
    cur="$(grep -E '^# [0-9]+/' "$f" 2>/dev/null | tail -1 | sed 's/^# //')"
    ep="$(grep -c '=== EPOCH' "$f" 2>/dev/null | head -1)"
    ult="$(grep -E '=== EPOCH' "$f" 2>/dev/null | tail -1)"
    mio="$(grep -E '^mIoU' "$f" 2>/dev/null | tail -1)"
    echo "  gpu $g: ${n:-0}/${t:-?} terminadas | ${ep:-0} epocas | en curso: ${cur:-?}"
    [ -n "$ult" ] && echo "          $ult   $mio"
    docker ps --filter name=sam2_g$g --format '{{.Names}}' | grep -q . && VIVOS=$((VIVOS+1))
  done
  echo "  contenedores vivos: $VIVOS"

  echo; echo "== combinaciones terminadas =="
  cat $P/logs/par_${TSX}_g*.log 2>/dev/null | grep -E '^>>> ' \
    | sed 's/^>>> //' | sort -t: -k2 -rn | sed 's/^/  /' || echo "  (ninguna)"

  if [ "$(cat $P/logs/par_${TSX}_g*.log 2>/dev/null | grep -c '=== EPOCH')" = "0" ]; then
    echo; echo "== NINGUNA EPOCA TODAVIA: cola del primer log =="
    tail -30 "$(ls $P/logs/par_${TSX}_g*.log | head -1)"
  fi
  for f in $P/logs/par_${TSX}_g*.log; do
    grep -q 'RESULTADOS' "$f" 2>/dev/null && { echo; echo "-- $(basename $f) --"; sed -n '/^RESULTADOS/,$p' "$f"; }
  done
  exit 0
fi

L="$(ls -t $P/logs/run_*.log 2>/dev/null | head -1)"
if [ -z "$L" ]; then
  echo; echo "== sin log; salida del contenedor =="
  docker logs sam2stk 2>&1 | tail -30
  exit 0
fi

N=$(grep -c '=== EPOCH' "$L" || true)
echo; echo "log: $L   ($N epocas)"
if [ "$N" -gt 0 ]; then
  echo; awk '/=== EPOCH/{n++} n>=1' "$L" | tail -34
else
  echo; echo "== NO HAY EPOCAS: ultimas lineas del log =="
  tail -40 "$L"
  echo; echo "== salida del contenedor =="
  docker logs sam2stk 2>&1 | tail -25
fi
REMOTE
