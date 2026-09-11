#!/usr/bin/env bash
# Se ejecuta EN LOCAL:  STK_SERVER=usuario@host bash servidor/probar_sam2.sh
# Comprueba dentro del contenedor del servidor que sam2 instala, que el checkpoint
# descarga y que el encoder produce las shapes esperadas. Solo prueba, no entrena.
set -euo pipefail

SERVER="${STK_SERVER:?define STK_SERVER=usuario@host}"
REMOTE="${STK_REMOTE:-$(ssh "$SERVER" 'echo "$HOME/sam2_stk"')}"

ssh "$SERVER" "mkdir -p '$REMOTE'"
ssh "$SERVER" 'bash -s' <<REMOTESH
set -euo pipefail
P=$REMOTE
cd "\$P"

echo "══ GPU libre ══"
GPU=\$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | tr -d ' ' | sort -t, -k2 -rn | head -1 | cut -d, -f1)
[ -n "\$GPU" ] || { echo "ERROR: ninguna GPU con >20 GB libres" >&2; exit 1; }
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed 's/^/  /'
echo "  -> usando GPU \$GPU (la que mas memoria libre tiene)"

echo
echo "══ imagen con torch ══"
IMG=\$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -v '<none>' \
      | grep -iE 'vllm|pytorch|cuda' | head -1)
echo "  \$IMG"

docker rm -f sam2probe >/dev/null 2>&1 || true
docker run --rm --name sam2probe \
  --gpus all -e CUDA_VISIBLE_DEVICES="\$GPU" --ipc=host \
  -u "\$(id -u):\$(id -g)" -v "\$P:/work" -w /work \
  -e HOME=/work -e HF_HOME=/work/.hf -e PYTHONUSERBASE=/work/.pydeps \
  --entrypoint /bin/bash "\$IMG" -c '
set -e
echo
echo "══ torch del contenedor ══"
python3 -c "import torch;print(\"  torch\",torch.__version__,\"| cuda\",torch.cuda.is_available(),\"|\",torch.cuda.get_device_name(0))"

echo
echo "══ instalando sam2 ══"
if python3 -c "import sam2" 2>/dev/null; then
  echo "  ya instalado, se reutiliza"
else
  export SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1
  timeout 900 pip install --progress-bar off --user sam2 \
      || { echo "ERROR: pip fallo o supero los 15 min" >&2; exit 1; }
fi
python3 -c "import sam2, os; print(\"  sam2 desde\", os.path.dirname(sam2.__file__))"
python3 -c "import antlr4, omegaconf; print(\"  antlr4 y omegaconf compatibles\")"

echo
echo "══ descargando checkpoint y midiendo ══"
python3 - <<"PY"
import time, torch
from sam2.build_sam import build_sam2_hf
t0=time.time()
m = build_sam2_hf("facebook/sam2.1-hiera-small", device="cuda")
enc = m.image_encoder.eval()
print("  checkpoint cargado en %.1fs" % (time.time()-t0))
print("  params encoder: %.1f M" % (sum(p.numel() for p in enc.parameters())/1e6))

for s in [448, 512, 1024]:
    x = torch.randn(2,3,s,s, device="cuda")
    with torch.no_grad():
        enc(x); torch.cuda.synchronize(); t=time.time()
        for _ in range(5): enc(x)
        torch.cuda.synchronize()
    o = enc(x)
    sh = [tuple(f.shape) for f in o["backbone_fpn"]]
    print("  %4d -> %6.1f ms/imagen   fpn %s" % (s, (time.time()-t)/5/2*1000, sh))
PY
'
REMOTESH
