#!/usr/bin/env bash
# Se ejecuta EN EL SERVIDOR. Reparte una rejilla entre las GPUs libres, un
# contenedor por GPU. No toca el contenedor 'sam2stk' de una corrida suelta.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
mkdir -p logs salidas experimentos/resultados

MIN_FREE="${STK_MIN_FREE_MB:-40000}"
if [ -n "${STK_GPUS:-}" ]; then
  GPUS="$STK_GPUS"
else
  echo "memoria libre por GPU:"
  GPUS=""
  while IFS=, read -r idx free; do
    idx="$(echo $idx | tr -d ' ')"; free="$(echo $free | tr -d ' ')"
    if [ "$free" -ge "$MIN_FREE" ]; then
      echo "  gpu $idx: ${free} MiB  -> LIBRE"; GPUS="$GPUS $idx"
    else
      echo "  gpu $idx: ${free} MiB  -> ocupada, se salta"
    fi
  done <<< "$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"
fi
GPUS="$(echo $GPUS)"
[ -n "$GPUS" ] || { echo "ERROR: ninguna GPU con >= ${MIN_FREE} MiB libres" >&2; exit 1; }
N=$(echo $GPUS | wc -w | tr -d ' ')
echo "gpus    : $GPUS  ($N particiones)"

IMG="${STK_IMAGE:-$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' \
      | grep -iE 'vllm|pytorch|cuda' | head -1)}"
echo "imagen  : $IMG"

TARGS=""
for a in "$@"; do TARGS="$TARGS $(printf '%q' "$a")"; done

TS="$(date +%Y%m%d_%H%M%S)"
i=0
for g in $GPUS; do
  NAME="sam2_g$g"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" \
    --gpus all -e CUDA_VISIBLE_DEVICES="$g" --ipc=host \
    -u "$(id -u):$(id -g)" -v "$PROJ:/work" -w /work \
    -e HOME=/work -e HF_HOME=/work/.hf -e PYTHONUSERBASE=/work/.pydeps \
    -e SAM2_BUILD_CUDA=0 -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint /bin/bash "$IMG" -c "
      set -o pipefail
      python3 experimentos/sweep.py --data-root /work/data \
        --shard $i --num-shards $N \
        --out-dir /work/salidas/barrido_${TS}_g$g \
        --runs-csv /work/experimentos/resultados/runs_${TS}_g$g.csv \
        --results-csv /work/experimentos/resultados/sweep_${TS}_g$g.csv$TARGS \
        2>&1 | tee /work/logs/par_${TS}_g$g.log
    " >/dev/null
  echo "  gpu $g -> $NAME (particion $i/$N)"
  i=$((i+1))
done

sleep 8
echo
docker ps --filter name=sam2_g --format '  {{.Names}}  {{.Status}}'
echo
echo "  progreso : bash servidor/estado.sh"
