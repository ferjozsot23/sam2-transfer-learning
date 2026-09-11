#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  STK_SERVER=usuario@host bash servidor/gpus.sh
# Memoria libre por GPU y quien las esta usando. Solo lee.
set -euo pipefail
ssh "${STK_SERVER:?define STK_SERVER=usuario@host}" 'bash -s' <<'REMOTE'
echo "== memoria libre =="
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
           --format=csv,noheader | sed 's/^/  gpu /'
echo
echo "== procesos =="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null \
  | head -20 | sed 's/^/  /' || echo "  (sin detalle de procesos)"
echo
echo "== nuestros contenedores =="
docker ps -a --filter name=sam2 --format '  {{.Names}}  {{.Status}}' 2>/dev/null || echo "  (ninguno)"
REMOTE
