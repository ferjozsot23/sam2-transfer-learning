"""Genera las figuras del README, en version clara y oscura.

Paleta validada para daltonismo y contraste. Se ejecuta desde cualquier sitio:
las rutas se resuelven contra la raiz del repositorio.
"""

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
                  serie='#2a78d6', serie2='#eb6834', grid='#e6e5e1'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', ink3='#7d7c74',
                 serie='#3987e5', serie2='#d95926', grid='#333330'),
}

IOU = [('background', 0.8595, 0.7768), ('track', 0.8572, 0.7365),
       ('kart', 0.7744, 0.6709), ('pickup', 0.6336, 0.5013),
       ('nitro', 0.4614, 0.5352), ('bomb', 0.1377, 0.0194),
       ('projectile', 0.0000, 0.0000)]

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
    fig.text(0.012, 0.905, 'mIoU 0.5320 frente a 0.4629 · entrenando el 3.6% del modelo',
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


if __name__ == '__main__':
    import csv

    filas = [r for r in csv.DictReader(open('experimentos/resultados/epocas.csv'))
             if r['notes'] == 'base_plus_s448_d256_lr0.0003_p0.20']
    mio = [float(r['val_miou']) for r in sorted(filas, key=lambda r: int(r['epoch']))]

    ejemplos = [('volcano_island_frame_0155', 'volcano_island'),
                ('lighthouse_frame_0120', 'lighthouse · escena nocturna'),
                ('volcano_island_frame_0125', 'volcano_island')]
    for modo in ('light', 'dark'):
        fig_comparacion(modo)
        fig_cualitativa(modo, ejemplos)
        print('figuras', modo)
