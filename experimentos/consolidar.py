"""Une los sweep_*.csv de todas las particiones y ordena por mIoU.

Conserva la mejor de las repeticiones de una misma configuracion y reporta el
rango entre ellas, porque esa diferencia es la medida directa de la varianza
entre corridas.

    python experimentos/consolidar.py
    python experimentos/consolidar.py --glob 'experimentos/resultados/sweep_*_g*.csv'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob as globmod

from utils import CLASS_NAMES

RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
CORTO = ['bg', 'trck', 'kart', 'pick', 'nitr', 'bomb', 'proj']


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default=os.path.join(RESULTADOS, 'sweep*.csv'))
    ap.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'salidas', 'consolidado.csv'))
    ap.add_argument('--top', type=int, default=0)
    args = ap.parse_args()

    ficheros = sorted(globmod.glob(args.glob))
    if not ficheros:
        consolidado = os.path.join(RESULTADOS, 'configuraciones.csv')
        if os.path.exists(consolidado):
            ficheros = [consolidado]
        else:
            raise SystemExit('no hay ficheros que casen con %r' % args.glob)

    filas = []
    for f in ficheros:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r['_origen'] = os.path.basename(f)
                filas.append(r)

    porcombo = {}
    for r in filas:
        porcombo.setdefault(r['combo'], []).append(r)
    repetidas = {k: v for k, v in porcombo.items() if len(v) > 1}
    unicas = [max(v, key=lambda r: num(r['best_miou'])) for v in porcombo.values()]

    ok = [r for r in unicas if num(r['best_miou']) == num(r['best_miou'])]
    ok.sort(key=lambda r: -num(r['best_miou']))
    if args.top:
        mostrar = ok[:args.top]
    else:
        mostrar = ok

    hdr = ('%-34s %-10s %5s %4s %8s %6s %7s ' %
           ('combinacion', 'backbone', 'size', 'dim', 'lr', 'power', 'mIoU'))
    hdr += ' '.join('%6s' % s for s in CORTO) + '  %4s %6s' % ('ep', 'min')
    print('=' * len(hdr))
    print('BARRIDO CONSOLIDADO — %d combinaciones de %d fichero(s)' % (len(unicas), len(ficheros)))
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for r in mostrar:
        linea = ('%-34s %-10s %5s %4s %8s %6s %7.4f ' %
                 (r['combo'][:34], r['backbone'], r['size'], r['dim'],
                  r['lr'], r['weight_power'], num(r['best_miou'])))
        linea += ' '.join('%6.4f' % num(r['iou_' + c]) if num(r['iou_' + c]) == num(r['iou_' + c])
                          else '   nan' for c in CLASS_NAMES)
        print(linea + '  %4s %6s' % (r['best_epoch'], r['minutos']))
    for r in unicas:
        if num(r['best_miou']) != num(r['best_miou']):
            print('%-34s  [%s]' % (r['combo'][:34], r.get('estado', '?')))

    if repetidas:
        print('-' * len(hdr))
        print('VARIANZA entre corridas de la MISMA configuracion:')
        for k, v in repetidas.items():
            vals = sorted(num(x['best_miou']) for x in v)
            print('  %-40s %s   (rango %.4f)'
                  % (k, ' / '.join('%.4f' % x for x in vals), vals[-1] - vals[0]))

    if ok:
        b = ok[0]
        print('-' * len(hdr))
        print('GANADORA: %s   mIoU %.4f  (epoca %s)' % (b['combo'], num(b['best_miou']), b['best_epoch']))
        print('  por clase: ' + ',  '.join('%s %.4f' % (c, num(b['iou_' + c])) for c in CLASS_NAMES))
        print('  modelo   : salidas/barrido_*/model_%s.th' % b['combo'])
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k in ok[0] if k != '_origen'],
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(ok)
        print('  tabla    -> %s' % args.out)

    print()
    print('EFECTO DE CADA EJE (mIoU medio de las combinaciones que lo comparten):')
    for eje in ['backbone', 'size', 'dim', 'lr', 'weight_power']:
        grupos = {}
        for r in ok:
            grupos.setdefault(r[eje], []).append(num(r['best_miou']))
        if len(grupos) < 2:
            continue
        orden = sorted(grupos.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        print('  %-14s %s' % (eje, '   '.join('%s=%.4f (n=%d)' % (k, sum(v) / len(v), len(v))
                                              for k, v in orden)))
    print('=' * len(hdr))


if __name__ == '__main__':
    main()
