#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  bash servidor/recoger.sh [--sin-modelos]
# Descarga metricas y logs primero, y despues (salvo --sin-modelos) los modelos.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER=fsotoj@172.28.230.10
REMOTE=/home/fsotoj/sam2_stk
MODELOS=1
[ "${1:-}" = "--sin-modelos" ] && MODELOS=0

CTL="$HOME/.ssh/cm_sam2_rec_$$"
mkdir -p "$HOME/.ssh"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$CTL" -o ControlPersist=10m)
trap '"${SSH[@]}" -O exit "$SERVER" 2>/dev/null || true' EXIT
export RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=10m"

mkdir -p logs salidas/resultados_crudos

traer() {
  local desc="$1"; shift
  if rsync -az --partial "$@" 2>/tmp/_rsync_err; then
    printf '  ok    %s\n' "$desc"
  else
    printf '  FALLO %s\n' "$desc"
    sed 's/^/        /' /tmp/_rsync_err | head -4
  fi
}

echo "== en el servidor: experimentos/resultados =="
"${SSH[@]}" "$SERVER" "ls -1 '$REMOTE/experimentos/resultados/'" 2>/dev/null | sed 's/^/  /' || echo "  (no se pudo listar)"
echo

traer "metricas" "$SERVER:$REMOTE/experimentos/resultados/" ./salidas/resultados_crudos/
traer "logs" "$SERVER:$REMOTE/logs/" ./logs/
if [ "$MODELOS" = "1" ]; then
  traer "modelos sueltos" "$SERVER:$REMOTE/salidas/"*.th ./salidas/
  traer "modelos del barrido" --relative "$SERVER:$REMOTE/./salidas/"barrido_* ./ \
        --exclude='ck_*/' --include='*/' --include='model_*.th' --exclude='*'
else
  echo "  (modelos omitidos por --sin-modelos)"
fi
