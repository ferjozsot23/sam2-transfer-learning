#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  STK_SERVER=usuario@host bash servidor/estado.sh
set -euo pipefail
ssh "${STK_SERVER:?define STK_SERVER=usuario@host}" "P='${STK_REMOTE:-}' bash -s" <<'REMOTE'
P="${P:-$HOME/sam2_stk}"
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
    seg="$(grep -E '^seg' "$f" 2>/dev/null | tail -3 | awk '{s+=$2; n++} END {if(n)printf "%.1f", s/n}')"
    echo "  gpu $g: ${n:-0}/${t:-?} terminadas | ${ep:-0} epocas | ${seg:-?} s/epoca | en curso: ${cur:-?}"
    [ -n "$ult" ] && echo "          $ult   $mio"
    docker ps --filter name=sam2_g$g --format '{{.Names}}' | grep -q . && VIVOS=$((VIVOS+1))
  done
  echo "  contenedores vivos: $VIVOS"

  SEGT="$(cat $P/logs/par_${TSX}_g*.log 2>/dev/null | grep -E '^seg' | tail -20 \
          | awk '{s+=$2; n++} END {if(n)printf "%.1f", s/n}')"
  HECHAS="$(cat $P/logs/par_${TSX}_g*.log 2>/dev/null | grep -c '^>>> ')"
  UNO="$(ls $P/logs/par_${TSX}_g*.log | head -1)"
  TOTAL="$(grep -m1 -oE '\(de [0-9]+\)' "$UNO" 2>/dev/null | tr -dc '0-9')"
  [ -z "$TOTAL" ] && TOTAL="$(grep -m1 -oE 'BARRIDO: [0-9]+' "$UNO" 2>/dev/null | tr -dc '0-9')"
  NG="$(ls $P/logs/par_${TSX}_g*.log 2>/dev/null | wc -l | tr -d ' ')"
  [ -n "$SEGT" ] && awk -v s="$SEGT" -v h="$HECHAS" -v t="${TOTAL:-0}" -v g="$NG" 'BEGIN{
      if (t>0 && g>0) {
        base = s*30/60;
        factor = 1.5;
        printf "  ritmo: %.1f s/epoca -> %.1f min la combinacion mas barata (none)\n", s, base;
        printf "  quedan %d de %d combinaciones\n", t-h, t;
        printf "  estimado total: ~%.0f min desde el arranque (%d gpus en paralelo)\n",
               t*base*factor/g, g;
      }}'

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
