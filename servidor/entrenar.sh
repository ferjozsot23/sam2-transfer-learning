#!/usr/bin/env bash
# Se ejecuta EN EL SERVIDOR. Localiza el dataset, resuelve el entorno dentro de
# un contenedor Docker y lanza el entrenamiento desacoplado de la sesion ssh.
#
#   bash servidor/entrenar.sh                       # configuracion por defecto
#   bash servidor/entrenar.sh --backbone base_plus  # los flags van a train.py
#
# Variables:
#   STK_GPU=3       forzar GPU (por defecto, la que mas memoria libre tenga)
#   STK_IMAGE=x:y   forzar imagen de Docker
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
mkdir -p logs salidas experimentos/resultados

DATA="$PROJ/data"
[ -d "$DATA/masks" ] || { echo "ERROR: no encuentro $DATA/masks" >&2; exit 1; }
echo "dataset : $DATA  ($(ls "$DATA"/masks | wc -l | tr -d ' ') mascaras)"

GPU="${STK_GPU:-}"
if [ -z "$GPU" ]; then
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed 's/^/  gpu /'
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | sort -t, -k2 -rn | head -1 | cut -d, -f1)
fi
echo "gpu     : $GPU"

IMG="${STK_IMAGE:-}"
if [ -z "$IMG" ]; then
  IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' \
        | grep -iE 'vllm|pytorch|cuda' | head -1)
fi
echo "imagen  : $IMG"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$PROJ/logs/run_$TS.log"
NAME="sam2stk"

TARGS=""
for a in "$@"; do TARGS="$TARGS $(printf '%q' "$a")"; done

docker rm -f "$NAME" >/dev/null 2>&1 || true
if ! docker run -d --name "$NAME" \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" --ipc=host \
  -u "$(id -u):$(id -g)" -v "$PROJ:/work" -w /work \
  -e HOME=/work -e HF_HOME=/work/.hf -e PYTHONUSERBASE=/work/.pydeps \
  -e SAM2_BUILD_CUDA=0 -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /bin/bash "$IMG" -c "
    python3 -c 'import sam2' 2>/dev/null || pip install --progress-bar off --user sam2
    python3 -c \"import torch,sam2;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| sam2 ok')\"
    set -o pipefail
    python3 train.py --data-root /work/data --out /work/salidas/model.th \
      --ckpt-dir /work/salidas/checkpoints \
      --runs-csv /work/experimentos/resultados/runs.csv$TARGS 2>&1 | tee /work/logs/run_$TS.log
  "
then
  echo >&2
  echo "ERROR: docker run fallo. Revisa el runtime de NVIDIA y el grupo docker." >&2
  exit 1
fi

sleep 8
echo
echo "lanzado en el contenedor '$NAME'"
echo "  log    : $LOG"
echo "  seguir : docker logs -f $NAME"
echo "  parar  : docker rm -f $NAME"
echo
docker logs "$NAME" 2>&1 | tail -20
