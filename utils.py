"""Carga, emparejamiento y exploracion del dataset de SuperTuxKart.

El dataset es plano: images/<track>_frame_<id>.png y su mascara en
masks/<track>_mask_combined_<id>.png. El emparejamiento se hace por la tupla
(track, id) porque el id no es unico entre tracks.

Dependencias: torch, numpy, Pillow.
"""

import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ['background', 'track', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
NUM_CLASSES = 7

VAL_TRACKS = ['volcano_island', 'lighthouse']

PALETTE = np.array([
    [  0,   0,   0],
    [128, 128, 128],
    [220,  40,  40],
    [ 40, 180,  90],
    [ 60, 130, 250],
    [250, 190,  40],
    [230,  60, 230],
], dtype=np.uint8)

_MASK_RE = re.compile(r'^(?P<track>[^.].*)_mask_combined_(?P<id>\d+)\.png$')


def _es_basura(name):
    return name.startswith('.') or name.startswith('__MACOSX')


def list_tracks(data_root):
    tracks = set()
    for name in os.listdir(os.path.join(data_root, 'masks')):
        if _es_basura(name):
            continue
        m = _MASK_RE.match(name)
        if m:
            tracks.add(m.group('track'))
    return sorted(tracks)


def list_samples(data_root, tracks):
    tracks = set(tracks)
    images = os.path.join(data_root, 'images')
    masks = os.path.join(data_root, 'masks')
    samples = []
    for name in sorted(os.listdir(masks)):
        if _es_basura(name):
            continue
        m = _MASK_RE.match(name)
        if m is None or m.group('track') not in tracks:
            continue
        frame = os.path.join(images, '%s_frame_%s.png' % (m.group('track'), m.group('id')))
        if os.path.exists(frame):
            samples.append((frame, os.path.join(masks, name)))
    return samples


def split_tracks(data_root, split):
    all_tracks = list_tracks(data_root)
    if split == 'all':
        return all_tracks
    val = [t for t in all_tracks if t in VAL_TRACKS]
    if split == 'val':
        return val
    if split == 'train':
        return [t for t in all_tracks if t not in VAL_TRACKS]
    raise ValueError("split debe ser 'train', 'val' o 'all'; llego %r" % split)


class STKSegmentationDataset(Dataset):
    def __init__(self, data_root, tracks, size=(448, 448), augment=False):
        self.data_root = data_root
        self.tracks = list(tracks)
        self.size = tuple(size) if size is not None else None
        self.augment = augment
        self.samples = list_samples(data_root, self.tracks)
        if not self.samples:
            raise RuntimeError('0 muestras para tracks=%r en %s' % (tracks, data_root))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, mask_path = self.samples[idx]
        img = Image.open(frame_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.size is not None:
            h, w = self.size
            img = img.resize((w, h), Image.BILINEAR)
            mask = mask.resize((w, h), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_np = np.asarray(img, dtype=np.uint8)
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        img_t = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float().div_(255.0)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))
        return img_t, mask_t

    def _augment(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
        return img, mask


def load_data(data_root, split='train', size=(448, 448), batch_size=8,
              num_workers=4, augment=False, limit=None, shuffle=None):
    dataset = STKSegmentationDataset(data_root, split_tracks(data_root, split),
                                     size=size, augment=augment)
    if limit is not None and limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(limit)))
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False,
                      pin_memory=torch.cuda.is_available())


def label_to_color(mask):
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    return PALETTE[np.clip(np.asarray(mask), 0, NUM_CLASSES - 1)]


def overlay(img_chw, mask_hw, alpha=0.55):
    img = (np.asarray(img_chw).transpose(1, 2, 0) * 255).astype(np.uint8)
    return (img * (1 - alpha) + label_to_color(mask_hw) * alpha).astype(np.uint8)


class ConfusionMatrix(object):
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self):
        self.matrix.zero_()

    def add(self, preds, labels):
        preds = preds.detach().reshape(-1).cpu()
        labels = labels.detach().reshape(-1).cpu()
        k = (labels >= 0) & (labels < self.num_classes)
        idx = self.num_classes * labels[k].to(torch.int64) + preds[k].to(torch.int64)
        self.matrix += torch.bincount(idx, minlength=self.num_classes ** 2) \
                            .reshape(self.num_classes, self.num_classes)

    @property
    def class_iou(self):
        m = self.matrix.double()
        tp = m.diag()
        denom = m.sum(dim=1) + m.sum(dim=0) - tp
        iou = torch.full((self.num_classes,), float('nan'), dtype=torch.float64)
        present = denom > 0
        iou[present] = tp[present] / denom[present]
        return iou.float()

    @property
    def miou(self):
        iou = self.class_iou
        valid = ~torch.isnan(iou)
        if valid.sum() == 0:
            return float('nan')
        return float(iou[valid].mean())

    @property
    def num_present(self):
        return int((~torch.isnan(self.class_iou)).sum())

    @property
    def global_accuracy(self):
        total = self.matrix.sum()
        if total == 0:
            return float('nan')
        return float(self.matrix.diag().sum().double() / total.double())


def format_iou_table(class_iou, miou, extra=None, title='VAL'):
    lines = ['=== %s ===' % title]
    for name, v in zip(CLASS_NAMES, list(class_iou)):
        v = float(v)
        lines.append('%-12s %s' % (name, 'nan' if v != v else '%.4f' % v))
    lines.append('-' * 25)
    present = int(sum(1 for v in class_iou if float(v) == float(v)))
    lines.append('%-12s %.4f   (%d clases presentes)' % ('mIoU', miou, present)
                 if miou == miou else '%-12s nan' % 'mIoU')
    for k, v in (extra or {}).items():
        lines.append('%-12s %s' % (k, v))
    lines.append('=' * 25)
    return '\n'.join(lines)


def format_confusion(cm, top=3, min_pixels=1):
    m = cm.matrix.double()
    lines = ['--- a donde van los pixeles de cada clase real ---']
    for c, name in enumerate(CLASS_NAMES):
        total = float(m[c].sum())
        if total < min_pixels:
            lines.append('%-12s (no aparece en validacion)' % name)
            continue
        order = sorted(range(cm.num_classes), key=lambda j: -float(m[c, j]))[:top]
        parts = ['%s %.1f%%' % (CLASS_NAMES[j], 100.0 * float(m[c, j]) / total)
                 for j in order if float(m[c, j]) > 0]
        lines.append('%-12s %10d px  ->  %s' % (name, int(total), ',  '.join(parts)))
    return '\n'.join(lines)


def weights_from_counts(counts, power=0.30, clip=10.0):
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / counts.sum()
    w = (1.0 / (freq + 1e-9)) ** power
    w = w / w.mean()
    return np.clip(w, 0, clip)


def compute_class_weights(dataset, cache_path='class_weights.json', verbose=True,
                          power=0.30, clip=10.0):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            data = json.load(fh)
        w = weights_from_counts(data['pixel_counts'], power, clip)
        if verbose:
            print('class_weights de %s (power=%.2f, clip=%.1f)' % (cache_path, power, clip))
        return torch.tensor(w, dtype=torch.float32)

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(len(dataset)):
        _, mask = dataset[i]
        counts += np.bincount(mask.numpy().ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        if verbose and (i + 1) % 200 == 0:
            print('  %d/%d frames contados' % (i + 1, len(dataset)))

    freq = counts / counts.sum()
    w = weights_from_counts(counts, power, clip)
    if cache_path:
        with open(cache_path, 'w') as fh:
            json.dump({'class_names': CLASS_NAMES,
                       'pixel_counts': counts.tolist(),
                       'pixel_pct': (100 * freq).round(6).tolist(),
                       'weights_power_0.30': w.round(6).tolist()}, fh, indent=2)
        if verbose:
            print('conteo de pixeles escrito en %s' % cache_path)
    return torch.tensor(w, dtype=torch.float32)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--size', type=int, nargs=2, default=[448, 448])
    ap.add_argument('--out', default='salidas/sanity_pairs.png')
    args = ap.parse_args()

    train_tracks = split_tracks(args.data_root, 'train')
    val_tracks = split_tracks(args.data_root, 'val')
    print('tracks train   ->', train_tracks)
    print('tracks val     ->', val_tracks)
    assert not (set(train_tracks) & set(val_tracks)), 'tracks compartidos entre splits'

    train_ds = STKSegmentationDataset(args.data_root, train_tracks, size=args.size)
    val_ds = STKSegmentationDataset(args.data_root, val_tracks, size=args.size)
    print('len(train_ds)  ->', len(train_ds))
    print('len(val_ds)    ->', len(val_ds))
    assert len(train_ds) > 0 and len(val_ds) > 0

    img, mask = train_ds[0]
    print('img.shape      ->', img.shape)
    print('img.dtype      ->', img.dtype)
    print('img.min/max    -> %.4f / %.4f' % (img.min(), img.max()))
    print('mask.shape     ->', mask.shape)
    print('mask.dtype     ->', mask.dtype)
    print('mask.unique()  ->', mask.unique().tolist())
    assert img.dtype == torch.float32 and img.shape[0] == 3
    assert mask.dtype == torch.int64 and mask.shape == img.shape[1:]

    seen = set()
    for ds in (train_ds, val_ds):
        for i in random.Random(0).sample(range(len(ds)), 200):
            seen.update(ds[i][1].unique().tolist())
    print('clases en 400 muestras ->', sorted(seen))
    assert seen <= set(range(NUM_CLASSES)), 'etiqueta fuera de [0,6]: %r' % sorted(seen)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    picks = random.Random(1).sample(range(len(train_ds)), 5)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    for col, i in enumerate(picks):
        im, mk = train_ds[i]
        nombre = os.path.basename(train_ds.samples[i][0])
        axes[0, col].imshow(im.permute(1, 2, 0).numpy())
        axes[0, col].set_title(nombre, fontsize=8)
        axes[1, col].imshow(label_to_color(mk.numpy()))
        axes[1, col].set_title('clases: %s' % mk.unique().tolist(), fontsize=8)
        axes[2, col].imshow(overlay(im.numpy(), mk.numpy()))
        axes[2, col].set_title('overlay', fontsize=8)
        for row in range(3):
            axes[row, col].axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print('grilla de sanidad ->', args.out)
