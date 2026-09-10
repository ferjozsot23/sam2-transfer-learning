"""Genera las figuras del README, en version clara y oscura.

Paleta validada para daltonismo y contraste. Se ejecuta desde cualquier sitio:
las rutas se resuelven contra la raiz del repositorio.
"""

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from models import load_model
from utils import CLASS_NAMES, PALETTE, label_to_color, overlay

TH = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', ink3='#8a8984',
                  serie='#2a78d6', serie2='#eb6834', serie3='#1baf7a', grid='#e6e5e1'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', ink3='#7d7c74',
                 serie='#3987e5', serie2='#d95926', serie3='#199e70', grid='#333330'),
}

IOU = [('background', 0.8791, 0.7768), ('track', 0.8743, 0.7365),
       ('kart', 0.8158, 0.6709), ('pickup', 0.5496, 0.5013),
       ('nitro', 0.4562, 0.5352), ('bomb', 0.2180, 0.0194),
       ('projectile', 0.4124, 0.0000)]

FRAMES = {'background': 1000, 'track': 1000, 'kart': 1000, 'pickup': 433,
          'nitro': 394, 'bomb': 168, 'projectile': 1}


def estilo(fig, axes, t):
    fig.patch.set_facecolor(t['surface'])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(t['surface'])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=t['ink2'], labelsize=9, length=0)


def fig_comparacion(modo):
    t = TH[modo]
    nombres = [n for n, _, _ in IOU][::-1]
    sam = [a for _, a, _ in IOU][::-1]
    unet = [b for _, _, b in IOU][::-1]
    y = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.barh(y + 0.19, sam, height=0.34, color=t['serie'], zorder=3, label='SAM 2')
    ax.barh(y - 0.19, unet, height=0.34, color=t['ink3'], zorder=3, label='U-Net desde cero')
    ax.set_yticks(y)
    ax.set_yticklabels(nombres, color=t['ink'], fontsize=10)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for yi, a, n in zip(y, sam, nombres):
        ax.text(a + 0.015, yi + 0.19, '%.4f' % a, va='center', color=t['ink'], fontsize=9)
        nf = FRAMES[n]
        ax.text(1.16, yi, '%d frame%s' % (nf, '' if nf == 1 else 's'), va='center',
                ha='right', color=t['ink3'], fontsize=8.5)
    for yi, b in zip(y, unet):
        ax.text(b + 0.015, yi - 0.19, '%.4f' % b, va='center', color=t['ink2'], fontsize=9)
    ax.text(1.16, len(nombres) - 0.38, 'en entrenamiento', va='center', ha='right',
            color=t['ink3'], fontsize=8.5, style='italic')
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.09), ncol=2,
                    frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    estilo(fig, ax, t)
    ax.tick_params(axis='y', labelcolor=t['ink'])
    fig.suptitle('IoU por clase — validación en 2 circuitos nunca vistos',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.905, 'mIoU 0.6008 frente a 0.4629 · encoder de SAM 2 afinado entero',
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0.06, 1, 0.87])
    fig.savefig('docs/figuras/iou-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_cualitativa(modo, elegidos):
    t = TH[modo]
    modelo = load_model('model.th')
    fig, axes = plt.subplots(len(elegidos), 3, figsize=(9.6, 3.3 * len(elegidos)))
    for r, (nombre, etiqueta) in enumerate(elegidos):
        fp = 'data/images/%s.png' % nombre
        mp = 'data/masks/%s.png' % nombre.replace('_frame_', '_mask_combined_')
        img = Image.open(fp).convert('RGB')
        x = torch.from_numpy(np.asarray(img, np.uint8).copy()) \
                 .permute(2, 0, 1).float().div_(255)[None]
        with torch.no_grad():
            pred = modelo(x).argmax(1)[0].numpy()
        gt = np.asarray(Image.open(mp))
        for c, (dato, titulo) in enumerate([(np.asarray(img), 'entrada'),
                                            (label_to_color(pred), 'predicción'),
                                            (label_to_color(gt), 'verdad')]):
            ax = axes[r, c]
            ax.imshow(dato)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(t['grid']); s.set_linewidth(1)
            if r == 0:
                ax.set_title(titulo, color=t['ink'], fontsize=11.5, pad=8)
        axes[r, 0].set_ylabel(etiqueta, color=t['ink2'], fontsize=10, labelpad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[i] / 255.0, ec=t['grid'], lw=0.5)
               for i in range(7)]
    fig.legend(handles, CLASS_NAMES, loc='lower center', ncol=7, frameon=False,
               fontsize=9.5, labelcolor=t['ink2'], bbox_to_anchor=(0.5, 0.004))
    fig.patch.set_facecolor(t['surface'])
    for ax in axes.ravel():
        ax.set_facecolor(t['surface'])
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig('docs/figuras/cualitativo-%s.png' % modo, dpi=145, facecolor=t['surface'])
    plt.close(fig)


def fig_descongelado(modo):
    import csv
    import statistics as st

    t = TH[modo]
    filas = [r for r in csv.DictReader(open('experimentos/resultados/configuraciones.csv'))
             if r['lote'] == '20260909_204201' and r['estado'] == 'ok']
    orden = [('none', 'ninguno', '0%'), ('neck', 'cuello FPN', '0.3%'),
             ('b3', '3 bloques', '23%'), ('b15', '15 bloques', '50%'),
             ('b28', '28 bloques', '75%'), ('b48', '48 bloques', '100%')]
    series = [('0.0001', '1e-4', t['serie'], -0.1), ('1e-05', '1e-5', t['serie2'], 0.0),
              ('1e-06', '1e-6', t['serie3'], 0.1)]
    x = np.arange(len(orden))
    ctrl = [float(r['top5']) for r in filas if r['modo'] == 'none']
    mc, sc = st.mean(ctrl), st.stdev(ctrl)

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axhspan(mc - sc, mc + sc, color=t['grid'], lw=0, zorder=0)
    ax.axhline(mc, color=t['ink3'], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(3.35, mc - sc - 0.017, 'congelado, 12 réplicas: %.3f ± %.3f' % (mc, sc),
            color=t['ink2'], fontsize=9)
    finales = {}
    for el, etiqueta, color, dx in series:
        med, dev = [], []
        for mo, _, _ in orden:
            v = [float(r['top5']) for r in filas if r['modo'] == mo and r['encoder_lr'] == el]
            med.append(st.mean(v))
            dev.append(st.stdev(v))
        ax.errorbar(x + dx, med, yerr=dev, fmt='none', ecolor=color, elinewidth=1.5,
                    capsize=0, zorder=2)
        ax.plot(x + dx, med, color=color, lw=2, marker='o', ms=8, mec=t['surface'],
                mew=2, zorder=3, label='encoder-lr %s' % etiqueta)
        ax.text(x[-1] + 0.3, med[-1], etiqueta, color=t['ink'], fontsize=9.5, va='center')
        finales[el] = med[-1]
    mejor = finales['1e-05']
    ax.text(x[-1] - 0.05, mejor + 0.028, '+%.3f' % (mejor - mc), color=t['ink'],
            fontsize=10.5, ha='right', fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n%s' % (n, pc) for _, n, pc in orden], color=t['ink'], fontsize=9.5)
    ax.set_xlim(-0.5, len(orden) - 0.4)
    ax.set_ylim(0.42, 0.63)
    ax.set_ylabel('mIoU de validación (top5)', color=t['ink2'], fontsize=9.5)
    ax.yaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    estilo(fig, ax, t)
    ax.tick_params(axis='x', labelcolor=t['ink'])
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=3,
                    frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    fig.suptitle('Descongelar el encoder solo compensa si es entero y despacio',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.905, 'media ± desviación de 4 repeticiones por punto · banda gris: encoder congelado',
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0.05, 1, 0.87])
    fig.savefig('docs/figuras/descongelado-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_validacion_cruzada(modo):
    import statistics as st

    t = TH[modo]
    filas = [r for r in csv.DictReader(open('experimentos/resultados/configuraciones.csv'))
             if r['backbone'] == 'large' and r['lote'] in ('20260909_204201', '20260910_095607')]
    pliegues = [('f1', 'f1 · donde se eligió', 'lighthouse · volcano_island'),
                ('f2', 'f2', 'gran_paradiso_island · hacienda'),
                ('f3', 'f3', 'abyss · olivermath')]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    deltas = []
    for i, (p, _, _) in enumerate(pliegues):
        base = [float(r['top5']) for r in filas if r['pliegue'] == p and r['modo'] == 'none']
        afin = [float(r['top5']) for r in filas if r['pliegue'] == p and r['modo'] == 'b48'
                and r['encoder_lr'] == '1e-05']
        for v, dx, color in [(base, -0.18, t['ink3']), (afin, 0.18, t['serie'])]:
            ax.scatter(i + dx + rng.uniform(-0.05, 0.05, len(v)), v, s=38, color=color,
                       alpha=0.6, lw=0, zorder=3)
            ax.plot([i + dx - 0.13, i + dx + 0.13], [st.mean(v)] * 2, color=color, lw=3,
                    solid_capstyle='round', zorder=4)
        deltas.append(st.mean(afin) - st.mean(base))
        ax.text(i, max(base + afin) + 0.014, '%+.3f' % deltas[-1], ha='center',
                color=t['ink'], fontsize=11, fontweight='medium')
    ax.set_xticks(range(len(pliegues)))
    ax.set_xticklabels(['%s\n%s' % (a, b) for _, a, b in pliegues], color=t['ink'], fontsize=9.5)
    ax.set_xlim(-0.6, len(pliegues) - 0.4)
    ax.set_ylim(0.42, 0.65)
    ax.set_ylabel('mIoU de validación (top5)', color=t['ink2'], fontsize=9.5)
    ax.yaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    estilo(fig, ax, t)
    ax.tick_params(axis='x', labelcolor=t['ink'])
    marca = lambda c: plt.Line2D([], [], color=c, lw=3, marker='o', ms=6, mec=c)
    leg = ax.legend([marca(t['ink3']), marca(t['serie'])],
                    ['encoder congelado', 'encoder afinado entero · elr 1e-5'],
                    loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2,
                    frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    fig.suptitle('Validación cruzada: la mejora se mantiene en los 3 pliegues, pero es menor',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.905, 'cada punto es una corrida · línea: media · mejora media %+.3f'
             % (sum(deltas) / len(deltas)), color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0.05, 1, 0.87])
    fig.savefig('docs/figuras/validacion-cruzada-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_entrenamiento_validacion(modo):
    t = TH[modo]
    curvas = {}
    for r in csv.DictReader(open('experimentos/resultados/epocas.csv')):
        if r['lote'] not in ('20260909_204201', '20260910_095607') or not r['val_loss']:
            continue
        n = r['notes']
        if '_b48_el1e-05_' in n:
            m = 'afinado'
        elif '_none_' in n:
            m = 'congelado'
        else:
            continue
        p = n[-2:] if n.endswith(('_f2', '_f3')) else 'f1'
        curvas.setdefault((p, m), {}).setdefault(n, []).append(
            (int(r['epoch']), float(r['train_loss']), float(r['val_loss'])))

    pliegues = [('f1', 'f1 · lighthouse · volcano_island'),
                ('f2', 'f2 · gran_paradiso_island · hacienda'),
                ('f3', 'f3 · abyss · olivermath')]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), sharey=True)
    tope = 0.0
    for ax, (p, titulo) in zip(axes, pliegues):
        for m, color in [('congelado', t['ink3']), ('afinado', t['serie'])]:
            corridas = [sorted(v) for v in curvas[(p, m)].values()]
            minimo = max(2, len(corridas) // 2)
            ep, tr, va = [], [], []
            e = 0
            while sum(len(c) > e for c in corridas) >= minimo:
                vivas = [c[e] for c in corridas if len(c) > e]
                ep.append(e)
                tr.append(sum(x[1] for x in vivas) / len(vivas))
                va.append(sum(x[2] for x in vivas) / len(vivas))
                e += 1
            tope = max(tope, max(va), max(tr))
            ax.plot(ep, va, color=color, lw=2.2, zorder=3)
            ax.plot(ep, tr, color=color, lw=1.6, ls=(0, (4, 2.5)), zorder=3)
        ax.set_title(titulo, color=t['ink'], fontsize=10, loc='left')
        ax.set_xlabel('época', color=t['ink2'], fontsize=9.5)
        ax.yaxis.grid(True, color=t['grid'], lw=1, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, tope * 1.06)
    axes[0].set_ylabel('CrossEntropy ponderada', color=t['ink2'], fontsize=9.5)
    estilo(fig, axes, t)
    linea = lambda c, ls: plt.Line2D([], [], color=c, lw=2, ls=ls)
    leg = fig.legend([linea(t['ink3'], '-'), linea(t['serie'], '-'),
                      linea(t['ink2'], '-'), linea(t['ink2'], (0, (4, 2.5)))],
                     ['encoder congelado', 'encoder afinado entero · elr 1e-5',
                      'validación', 'entrenamiento'],
                     loc='lower center', ncol=4, frameon=False, fontsize=9.5,
                     bbox_to_anchor=(0.5, 0.0))
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    fig.suptitle('Entrenamiento vs validación: la pérdida de validación deja de bajar en las primeras épocas',
                 color=t['ink'], fontsize=12.5, x=0.008, ha='left', y=0.975)
    fig.text(0.008, 0.895, 'media por época de las corridas de cada pliegue · '
             'el modelo se elige por mIoU de validación, no por pérdida',
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0.07, 1, 0.87])
    fig.savefig('docs/figuras/entrenamiento-validacion-%s.png' % modo, dpi=170,
                facecolor=t['surface'])
    plt.close(fig)


if __name__ == '__main__':
    ejemplos = [('volcano_island_frame_0155', 'volcano_island'),
                ('lighthouse_frame_0120', 'lighthouse · escena nocturna'),
                ('volcano_island_frame_0125', 'volcano_island')]
    for modo in ('light', 'dark'):
        fig_comparacion(modo)
        fig_descongelado(modo)
        fig_validacion_cruzada(modo)
        fig_entrenamiento_validacion(modo)
        fig_cualitativa(modo, ejemplos)
        print('figuras', modo)
