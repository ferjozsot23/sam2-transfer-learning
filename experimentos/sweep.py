"""Barrido de hiperparametros: recorre una rejilla y consolida los resultados.

Con --num-shards N y --shard I procesa solo una de cada N combinaciones, lo que
permite repartir la misma rejilla entre varias GPUs lanzando N procesos. El
reparto es intercalado para que las combinaciones caras no se amontonen en una.

Una combinacion que falle no tumba el barrido: se registra y sigue.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import csv
import itertools
import time
import traceback

import torch

from train import train
from utils import CLASS_NAMES, PLIEGUES

def parse_modo(m):
    if m == 'none':
        return (0, False)
    if m == 'neck':
        return (0, True)
    if m.endswith('solo') and m[1:-4].isdigit():
        return (int(m[1:-4]), False)
    if m.startswith('b') and m[1:].isdigit():
        return (int(m[1:]), True)
    raise ValueError("modo %r no valido: usa none, neck, bN o bNsolo" % m)

CAMPOS = (['combo', 'backbone', 'size', 'dim', 'lr', 'weight_power', 'modo',
           'encoder_lr', 'rep', 'pliegue', 'augment', 'best_epoch', 'best_miou', 'top5', 'sigma']
          + ['iou_' + c for c in CLASS_NAMES] + ['minutos', 'estado'])


def construir(base, backbone, size, dim, lr, power, modo, elr, pliegue, tag):
    a = copy.deepcopy(base)
    a.backbone, a.size, a.dim, a.lr, a.weight_power = backbone, size, dim, lr, power
    a.unfreeze_blocks, a.unfreeze_neck = parse_modo(modo)
    a.encoder_lr = elr
    a.val_tracks = PLIEGUES[pliegue]
    a.notes = tag
    a.out = os.path.join(base.out_dir, 'model_%s.th' % tag)
    a.ckpt_dir = os.path.join(base.out_dir, 'ck_%s' % tag)
    a.resume = None
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--backbones', nargs='+', default=['small'])
    ap.add_argument('--sizes', type=int, nargs='+', default=[448])
    ap.add_argument('--dims', type=int, nargs='+', default=[128])
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3])
    ap.add_argument('--powers', type=float, nargs='+', default=[0.30])
    ap.add_argument('--modos', nargs='+', default=['none'],
                    help='none=congelado, neck=solo cuello, bN=N bloques finales + cuello, '
                         'bNsolo=N bloques sin cuello')
    ap.add_argument('--encoder-lrs', type=float, nargs='+', default=[1e-4])
    ap.add_argument('--repeticiones', type=int, default=1,
                    help='repite la rejilla N veces para medir la varianza entre corridas')
    ap.add_argument('--pliegues', nargs='+', default=['f1'], choices=sorted(PLIEGUES),
                    help='particiones de validacion por circuito')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--early-stop', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--weight-clip', type=float, default=10.0)
    ap.add_argument('--weights', default='class_weights.json')
    ap.add_argument('--no-weights', action='store_true')
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--unfreeze-blocks', type=int, default=0)
    ap.add_argument('--unfreeze-neck', action='store_true')
    ap.add_argument('--encoder-lr', type=float, default=1e-4)

    ap.add_argument('--device', default='auto')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out-dir', default='salidas/barrido')
    ap.add_argument('--runs-csv', default='experimentos/resultados/runs.csv')
    ap.add_argument('--results-csv', default='experimentos/resultados/sweep.csv')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    for m in args.modos:
        parse_modo(m)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
    rejilla = list(itertools.product(args.backbones, args.sizes, args.dims,
                                     args.lrs, args.powers, args.modos,
                                     args.encoder_lrs, range(1, args.repeticiones + 1), args.pliegues))
    total = len(rejilla)
    if args.num_shards > 1:
        rejilla = rejilla[args.shard::args.num_shards]

    print('=' * 78)
    print('BARRIDO: particion %d de %d -> %d combinaciones (de %d)'
          % (args.shard, args.num_shards, len(rejilla), total)
          if args.num_shards > 1 else 'BARRIDO: %d combinaciones' % total)
    for i, c in enumerate(rejilla, 1):
        print('  %2d/%d  backbone=%s size=%d dim=%d lr=%.0e power=%.2f modo=%s elr=%.0e rep=%d pliegue=%s'
              % ((i, len(rejilla)) + c))
    print('=' * 78, flush=True)
    if args.dry_run:
        return

    filas = []
    for i, (bb, sz, dm, lr, pw, mo, el, rep, pl) in enumerate(rejilla, 1):
        tag = '%s_s%d_d%d_lr%g_p%.2f_%s_el%g_r%d_%s' % (bb, sz, dm, lr, pw, mo, el, rep, pl)
        print('\n' + '#' * 78)
        print('# %d/%d  %s' % (i, len(rejilla), tag))
        print('#' * 78, flush=True)
        t0 = time.time()
        try:
            r = train(construir(args, bb, sz, dm, lr, pw, mo, el, pl, tag))
            estado = 'ok'
        except Exception:
            traceback.print_exc()
            r = {'best_miou': float('nan'), 'best_epoch': -1,
                 'top5': float('nan'), 'sigma': float('nan'),
                 'class_iou': [float('nan')] * len(CLASS_NAMES)}
            estado = 'ERROR'
        mins = (time.time() - t0) / 60.0
        filas.append(dict(zip(CAMPOS,
            [tag, bb, sz, dm, lr, pw, mo, el, rep, pl, int(not args.no_augment),
             r['best_epoch'], r['best_miou'], r['top5'], r['sigma']]
            + list(r['class_iou']) + [round(mins, 2), estado])))
        with open(args.results_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows(filas)
        print('\n>>> %s : top5 %.4f pico %.4f (epoca %s, %.1f min) [%s]'
              % (tag, r['top5'], r['best_miou'], r['best_epoch'], mins, estado), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = [f for f in filas if f['best_miou'] == f['best_miou']]
    ok.sort(key=lambda f: -(f['top5'] if f['top5'] == f['top5'] else f['best_miou']))
    corto = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']
    hdr = ('%-44s %7s %7s %7s ' % ('combinacion', 'top5', 'pico', 'sigma')
           + ' '.join('%6s' % s for s in corto) + '  %4s' % 'ep')
    print('\n\n' + '=' * len(hdr))
    print('RESULTADOS')
    print('=' * len(hdr)); print(hdr); print('-' * len(hdr))
    for f in ok:
        print('%-44s %7.4f %7.4f %7.4f ' % (f['combo'][:44], f['top5'], f['best_miou'], f['sigma'])
              + ' '.join('%6.4f' % f['iou_' + c] if f['iou_' + c] == f['iou_' + c]
                         else '   nan' for c in CLASS_NAMES)
              + '  %4s' % f['best_epoch'])
    for f in filas:
        if f['best_miou'] != f['best_miou']:
            print('%-44s  [%s]' % (f['combo'][:44], f['estado']))
    if ok:
        print('-' * len(hdr))
        print('GANADORA por top5: %s  ->  top5 %.4f (pico %.4f)'
              % (ok[0]['combo'], ok[0]['top5'], ok[0]['best_miou']))
    print('=' * len(hdr))


if __name__ == '__main__':
    main()
