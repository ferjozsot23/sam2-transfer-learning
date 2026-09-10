#!/usr/bin/env python3
"""Segmenta imagenes con el modelo entrenado.

    python predict.py imagen.png
    python predict.py carpeta/ --out resultados/ --masks

El .th guarda solo la cabeza (unos 3 MB): el encoder de SAM 2 se reconstruye al
cargar, descargando su checkpoint la primera vez. Necesita el paquete sam2.
"""

import argparse
import os
import sys

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def main():
    ap = argparse.ArgumentParser(
        description='Segmentacion semantica de SuperTuxKart en 7 clases.',
        epilog='Ejemplo:  python predict.py ejemplos/ --out salida/')
    ap.add_argument('entrada', help='imagen o carpeta de imagenes')
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', default='model.th')
    ap.add_argument('--masks', action='store_true',
                    help='guarda ademas la mascara cruda de clases (0..6, 1 canal)')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from models import load_model
    from utils import CLASS_NAMES, NUM_CLASSES, label_to_color, overlay

    if not os.path.exists(args.model):
        sys.exit("No se encuentra '%s'. Debe estar junto a models.py." % args.model)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    modelo = load_model(args.model, device=device)
    print('Modelo cargado en %s (backbone %s, entrada %d)'
          % (device, modelo.backbone, modelo.image_size))

    if os.path.isdir(args.entrada):
        files = sorted(os.path.join(args.entrada, f) for f in os.listdir(args.entrada)
                       if f.lower().endswith(IMG_EXT) and not f.startswith('.'))
    else:
        files = [args.entrada]
    if not files:
        sys.exit('No se encontraron imagenes en %s' % args.entrada)

    outdir = args.out or (args.entrada if os.path.isdir(args.entrada)
                          else os.path.dirname(os.path.abspath(args.entrada)))
    os.makedirs(outdir, exist_ok=True)

    for f in files:
        img = Image.open(f).convert('RGB')
        x = torch.from_numpy(np.asarray(img, np.uint8).copy()) \
                 .permute(2, 0, 1).float().div_(255.0)[None].to(device)
        with torch.no_grad():
            pred = modelo(x).argmax(1)[0].cpu().numpy().astype(np.uint8)

        stem = os.path.splitext(os.path.basename(f))[0]
        rgb = np.asarray(img)
        panel = np.concatenate([rgb, label_to_color(pred),
                                overlay(rgb.transpose(2, 0, 1) / 255.0, pred)], axis=1)
        out = os.path.join(outdir, stem + '_seg.png')
        Image.fromarray(panel).save(out)
        if args.masks:
            Image.fromarray(pred).save(os.path.join(outdir, stem + '_mask.png'))

        c = np.bincount(pred.ravel(), minlength=NUM_CLASSES)
        pct = 100.0 * c / c.sum()
        print('%-30s -> %s   %s' % (
            os.path.basename(f), os.path.basename(out),
            ' '.join('%s %.1f%%' % (CLASS_NAMES[i][:4], pct[i])
                     for i in range(NUM_CLASSES) if c[i] > 0)))

    print('\n%d imagen(es) en %s/' % (len(files), outdir.rstrip('/')))
    print('Panel: original | segmentacion | superposicion')


if __name__ == '__main__':
    main()
