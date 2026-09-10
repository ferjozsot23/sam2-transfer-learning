#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  bash servidor/recoger.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER=fsotoj@172.28.230.10
REMOTE=/home/fsotoj/sam2_stk
mkdir -p logs salidas experimentos/resultados
rsync -avz "$SERVER:$REMOTE/salidas/"*.th ./salidas/ 2>/dev/null || echo "(aun no hay modelos sueltos)"
rsync -avz --relative "$SERVER:$REMOTE/./salidas/"barrido_* ./ \
      --include='*/' --include='*.th' --exclude='*' 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/experimentos/resultados/" ./experimentos/resultados/ 2>/dev/null || true
rsync -avz "$SERVER:$REMOTE/logs/" ./logs/
echo
grep -B1 -A13 '=== EPOCH' logs/*.log 2>/dev/null | tail -20
