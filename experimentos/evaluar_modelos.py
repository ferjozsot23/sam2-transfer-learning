"""Mide uno o varios .th sobre el conjunto de validacion.

No se fia de lo que digan los csv del barrido: vuelve a calcular la matriz de
confusion sobre los 500 frames de validacion. Es la comprobacion que decide que
modelo se entrega.

    python experimentos/evaluar_modelos.py salidas/barrido_*/model_large*.th
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob

import torch

from models import load_model
from utils import CLASS_NAMES, ConfusionMatrix, load_data

CORTO = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']


@torch.no_grad()
def medir(path, data_root, size, device, batch_size):
    modelo = load_model(path, device=device)
    loader = load_data(data_root, 'val', size=(size, size), batch_size=batch_size,
                       num_workers=0, augment=False)
    cm = ConfusionMatrix()
    for img, mask in loader:
        cm.add(modelo(img.to(device)).argmax(1), mask.to(device))
    return cm, modelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('modelos', nargs='+')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')

    rutas = []
    for p in args.modelos:
        rutas.extend(sorted(glob.glob(p)) if any(c in p for c in '*?[') else [p])
    rutas = [p for p in rutas if os.path.isfile(p)]

    hdr = '%-46s %-9s %5s %7s ' % ('modelo', 'backbone', 'dim', 'mIoU')
    hdr += ' '.join('%6s' % s for s in CORTO)
    print(hdr)
    print('-' * len(hdr))

    filas = []
    for p in rutas:
        try:
            cm, m = medir(p, args.data_root, args.size, device, args.batch_size)
        except Exception as e:
            print('%-46s  ERROR: %s' % (os.path.basename(p)[:46], str(e)[:60]))
            continue
        iou = cm.class_iou
        filas.append((cm.miou, p, iou))
        print('%-46s %-9s %5d %7.4f ' % (os.path.basename(p)[:46], m.backbone, m.dim, cm.miou)
              + ' '.join('%6.4f' % float(v) if float(v) == float(v) else '   nan' for v in iou),
              flush=True)

    filas.sort(key=lambda f: -f[0])
    if filas:
        mi, p, iou = filas[0]
        print('-' * len(hdr))
        print('MEJOR MEDIDO: %s' % p)
        print('  mIoU %.4f' % mi)
        print('  ' + ',  '.join('%s %.4f' % (c, float(v)) for c, v in zip(CLASS_NAMES, iou)))


if __name__ == '__main__':
    main()
