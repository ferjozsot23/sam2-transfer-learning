#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  STK_SERVER=usuario@host bash servidor/lanzar.sh [flags de train.py]
# Sube el dataset la primera vez, sube el codigo y lanza el entrenamiento.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER="${STK_SERVER:?define STK_SERVER=usuario@host}"

CTL="$HOME/.ssh/cm_sam2_$$"
mkdir -p "$HOME/.ssh"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=10m)
trap '"${SSH[@]}" -O exit "$SERVER" 2>/dev/null || true' EXIT
export RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=10m"
REMOTE="${STK_REMOTE:-$("${SSH[@]}" "$SERVER" 'echo "$HOME/sam2_stk"')}"

"${SSH[@]}" "$SERVER" "mkdir -p '$REMOTE'"

if "${SSH[@]}" "$SERVER" "[ -d '$REMOTE/data/masks' ]"; then
  echo "==> el dataset ya esta en el servidor"
else
  echo "==> empaquetando el dataset (3001 archivos pequenos viajan mucho peor que uno grande)"
  COPYFILE_DISABLE=1 tar --no-xattrs -C "$(dirname "$(readlink data)")" \
      -czf /tmp/seg_dataset.tgz "$(basename "$(readlink data)")" 2>/dev/null
  echo "==> subiendo $(du -h /tmp/seg_dataset.tgz | cut -f1)"
  rsync -avz --progress /tmp/seg_dataset.tgz "$SERVER:$REMOTE/"
  "${SSH[@]}" "$SERVER" "cd '$REMOTE' && tar xzf seg_dataset.tgz && rm -f seg_dataset.tgz && \
      mv seg_dataset data 2>/dev/null || true && ls data | tr '\n' ' '"
  rm -f /tmp/seg_dataset.tgz
fi

echo "==> subiendo codigo"
rsync -avz --exclude data --exclude .venv --exclude salidas --exclude logs \
      --exclude .hf --exclude .pydeps --exclude __pycache__ --exclude '*.th' \
      --exclude .git --exclude .DS_Store ./ "$SERVER:$REMOTE/"

echo "==> lanzando"
LANZADOR=servidor/entrenar.sh
if [ "${1:-}" = "paralelo" ]; then LANZADOR=servidor/paralelo.sh; shift; fi
RARGS=""
for a in "$@"; do RARGS="$RARGS $(printf '%q' "$a")"; done

ENV_REMOTO=""
for v in STK_MIN_FREE_MB STK_GPUS STK_IMAGE STK_GPU STK_DATA_ROOT; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && ENV_REMOTO="$ENV_REMOTO $v=$(printf '%q' "$val")"
done
[ -n "$ENV_REMOTO" ] && echo "   reenviando:$ENV_REMOTO"

"${SSH[@]}" "$SERVER" "env$ENV_REMOTO bash '$REMOTE/$LANZADOR'$RARGS"
