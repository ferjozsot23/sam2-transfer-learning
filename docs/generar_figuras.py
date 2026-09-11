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

MODELO_FINAL = 'large_s448_d256_lr0.001_p0.30_b48_el1e-05_r4'

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
    fig.savefig('docs/figuras/descongelado-encoder-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_validacion_cruzada(modo):
    import statistics as st

    t = TH[modo]
    filas = [r for r in csv.DictReader(open('experimentos/resultados/configuraciones.csv'))
             if r['backbone'] == 'large' and r['lote'] in ('20260909_204201', '20260910_095607')]
    pliegues = [('f1', 'lighthouse\nvolcano_island'),
                ('f2', 'gran_paradiso_island\nhacienda'),
                ('f3', 'abyss\nolivermath')]
    mejoras = []
    for p, _ in pliegues:
        base = [float(r['top5']) for r in filas if r['pliegue'] == p and r['modo'] == 'none']
        afin = [float(r['top5']) for r in filas if r['pliegue'] == p and r['modo'] == 'b48'
                and r['encoder_lr'] == '1e-05']
        mejoras.append(st.mean(afin) - st.mean(base))
    media = st.mean(mejoras)

    x = np.arange(len(pliegues))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(x, mejoras, width=0.55, color=t['serie'], zorder=3)
    for xi, m in zip(x, mejoras):
        ax.text(xi, m + 0.0015, '+%.3f' % m, ha='center', va='bottom', color=t['ink'], fontsize=11)
    ax.axhline(0, color=t['ink3'], lw=1, zorder=4)
    ax.axhline(media, color=t['ink2'], lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(x[-1] + 0.9, media + max(mejoras) * 0.015, 'media +%.3f' % media, ha='right',
            va='bottom', color=t['ink2'], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n%s' % (p, c) for p, c in pliegues], color=t['ink'], fontsize=9.5)
    ax.set_xlim(-0.5, x[-1] + 0.95)
    ax.set_ylim(0, max(mejoras) * 1.22)
    ax.set_yticks([])
    estilo(fig, ax, t)
    ax.tick_params(axis='x', labelcolor=t['ink'])
    fig.suptitle('Afinar el encoder mejora el mIoU en los 3 pliegues',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.9, 'mejora frente al encoder congelado, según los circuitos de validación',
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig('docs/figuras/validacion-cruzada-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def curvas_modelo_final():
    filas = sorted((int(r['epoch']), float(r['train_loss']), float(r['val_loss']), float(r['val_miou']))
                   for r in csv.DictReader(open('experimentos/resultados/epocas.csv'))
                   if r['lote'] == '20260909_204201' and r['notes'] == MODELO_FINAL)
    return map(list, zip(*filas))


def fig_entrenamiento_validacion(modo):
    t = TH[modo]
    ep, tr, va, _ = curvas_modelo_final()

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(ep, tr, color=t['serie'], lw=2.2, zorder=3)
    ax.plot(ep, va, color=t['serie2'], lw=2.2, zorder=3)
    ax.text(ep[-1] + 0.4, tr[-1], 'entrenamiento', color=t['serie'], va='center', fontsize=10)
    ax.text(ep[-1] + 0.4, va[-1], 'validación', color=t['serie2'], va='center', fontsize=10)
    ax.set_xlim(-0.5, ep[-1] + 4.8)
    ax.set_ylim(0, max(va) * 1.05)
    ax.set_xlabel('época', color=t['ink2'], fontsize=9.5)
    ax.set_ylabel('pérdida', color=t['ink2'], fontsize=9.5)
    ax.yaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    estilo(fig, ax, t)
    fig.suptitle('Pérdida de entrenamiento vs validación',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.9, 'la de entrenamiento sigue bajando; la de validación se estanca desde la época %d'
             % ep[va.index(min(va))], color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig('docs/figuras/entrenamiento-validacion-%s.png' % modo, dpi=170,
                facecolor=t['surface'])
    plt.close(fig)


def fig_miou_validacion(modo):
    t = TH[modo]
    ep, _, _, mi = curvas_modelo_final()
    mejor = mi.index(max(mi))

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(ep, mi, color=t['serie'], lw=2.2, zorder=3)
    ax.axvline(ep[mejor], color=t['ink3'], lw=1, ls=(0, (4, 3)), zorder=2)
    ax.scatter([ep[mejor]], [mi[mejor]], s=70, color=t['serie'], edgecolor=t['surface'],
               linewidth=2, zorder=4)
    ax.text(ep[mejor] + 0.5, mi[mejor] + 0.008, 'época %d · %.3f' % (ep[mejor], mi[mejor]),
            color=t['ink'], fontsize=10.5, va='bottom')
    ax.set_xlim(-0.5, ep[-1] + 1.5)
    ax.set_ylim(min(mi) - 0.03, max(mi) + 0.06)
    ax.set_xlabel('época', color=t['ink2'], fontsize=9.5)
    ax.set_ylabel('mIoU de validación', color=t['ink2'], fontsize=9.5)
    ax.yaxis.grid(True, color=t['grid'], lw=1, zorder=0)
    ax.set_axisbelow(True)
    estilo(fig, ax, t)
    fig.suptitle('mIoU de validación del modelo final',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.9, 'sube hasta la época %d, que es la que se guarda' % ep[mejor],
             color=t['ink2'], fontsize=9.5, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig('docs/figuras/miou-validacion-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


def fig_seleccion(modo):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    t = TH[modo]
    filas = [r for r in csv.DictReader(open('experimentos/resultados/configuraciones.csv'))
             if r['lote'] == '20260909_204201']
    niveles = [('none', 'ninguno', '0%'), ('neck', 'cuello FPN', '0.3%'),
               ('b3', '3 bloques', '23%'), ('b15', '15 bloques', '50%'),
               ('b28', '28 bloques', '75%'), ('b48', '48 bloques', '100%')]
    elrs = [('0.0001', '1e-4'), ('1e-05', '1e-5'), ('1e-06', '1e-6')]
    valor = {(r['modo'], r['encoder_lr'], int(r['rep'])): float(r['best_miou']) for r in filas}
    rampa = ['#e3edf9', '#1d5bab'] if modo == 'light' else ['#24313f', '#8cc0f5']
    cmap = LinearSegmentedColormap.from_list('rampa', rampa)
    norm = Normalize(min(valor.values()), max(valor.values()))
    lado, hueco = 0.2, 0.04
    x0 = (1 - (4 * lado + 3 * hueco)) / 2

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    for i, (mo, _, _) in enumerate(niveles):
        y = len(niveles) - 1 - i
        for j, (el, _) in enumerate(elrs):
            for rep in range(1, 5):
                x = j + x0 + (rep - 1) * (lado + hueco)
                ax.add_patch(Rectangle((x, y + 0.2), lado, 0.6, color=cmap(norm(valor[(mo, el, rep)])),
                                       lw=0, zorder=2))
    for rep in range(1, 5):
        for j in range(len(elrs)):
            ax.text(j + x0 + (rep - 1) * (lado + hueco) + lado / 2, len(niveles) - 0.12, 'r%d' % rep,
                    ha='center', va='bottom', color=t['ink3'], fontsize=8)

    jg, yg = 1, 0
    xg = jg + x0 + 3 * (lado + hueco)
    ax.add_patch(Rectangle((jg + x0 - 0.04, yg + 0.12), 4 * lado + 3 * hueco + 0.08, 0.76,
                           fill=False, ec=t['ink2'], lw=1.2, zorder=3))
    ax.add_patch(Rectangle((xg, yg + 0.2), lado, 0.6, fill=False, ec=t['ink'], lw=2.4, zorder=4))
    ax.text(xg + lado / 2, yg - 0.04, 'modelo final · mIoU %.3f' % valor[('b48', '1e-05', 4)],
            ha='center', va='top', color=t['ink'], fontsize=10.5, fontweight='medium')

    ax.set_xlim(0, len(elrs))
    ax.set_ylim(-0.5, len(niveles) + 0.25)
    ax.set_yticks([len(niveles) - 1 - i + 0.5 for i in range(len(niveles))])
    ax.set_yticklabels(['%s · %s' % (n, pc) for _, n, pc in niveles], color=t['ink'], fontsize=10)
    ax.set_xticks([j + 0.5 for j in range(len(elrs))])
    ax.set_xticklabels(['encoder-lr %s' % e for _, e in elrs], color=t['ink'], fontsize=10)
    ax.xaxis.tick_top()
    estilo(fig, ax, t)
    ax.tick_params(axis='both', labelcolor=t['ink'], pad=14)

    barra = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal',
                         fraction=0.035, pad=0.04, aspect=40)
    barra.outline.set_visible(False)
    barra.ax.tick_params(colors=t['ink2'], labelsize=9, length=0)
    barra.set_label('mejor mIoU de validación de cada corrida', color=t['ink2'], fontsize=9.5)

    fig.suptitle('Cómo se eligió el modelo final',
                 color=t['ink'], fontsize=12.5, x=0.012, ha='left', y=0.975)
    fig.text(0.012, 0.915, '72 corridas: 6 niveles de descongelado × 3 learning rates del encoder × 4 '
             'repeticiones\nprimero la combinación con mejor media; dentro de ella, la mejor repetición',
             color=t['ink2'], fontsize=9.5, ha='left', va='top', linespacing=1.5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.1)
    fig.savefig('docs/figuras/seleccion-%s.png' % modo, dpi=170, facecolor=t['surface'])
    plt.close(fig)


if __name__ == '__main__':
    ejemplos = [('volcano_island_frame_0155', 'volcano_island'),
                ('lighthouse_frame_0120', 'lighthouse · escena nocturna'),
                ('volcano_island_frame_0125', 'volcano_island')]
    for modo in ('light', 'dark'):
        fig_comparacion(modo)
        fig_descongelado(modo)
        fig_seleccion(modo)
        fig_validacion_cruzada(modo)
        fig_entrenamiento_validacion(modo)
        fig_miou_validacion(modo)
        fig_cualitativa(modo, ejemplos)
        print('figuras', modo)
