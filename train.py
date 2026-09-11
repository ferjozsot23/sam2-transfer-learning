"""Loop de entrenamiento de la cabeza y, opcionalmente, del encoder de SAM 2.

Vive en un modulo aparte para que el notebook lo importe y lo ejecute, y para
que el servidor pueda lanzarlo como script sin depender de un kernel de Jupyter.

Guarda checkpoint cada epoca y soporta --resume, para retomar corridas
interrumpidas en un servidor compartido.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

from models import SAM2Segmenter, save_model
from utils import (CLASS_NAMES, NUM_CLASSES, ConfusionMatrix, STKSegmentationDataset,
                   compute_class_weights, format_confusion, format_iou_table,
                   load_data, split_tracks)


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES)
    total_loss, nb = 0.0, 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        total_loss += float(criterion(logits, mask))
        nb += 1
        cm.add(logits.argmax(dim=1), mask)
    return cm, total_loss / max(nb, 1)


def train(args):
    device = pick_device(args.device)
    val_tracks = getattr(args, 'val_tracks', None) or None
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.abspath(args.out)
    last_ckpt = os.path.join(ckpt_dir, 'last.th')

    print('device       :', device)
    print('torch        :', torch.__version__)
    print('data-root    :', os.path.abspath(args.data_root))
    print('tracks train :', split_tracks(args.data_root, 'train', val_tracks))
    print('tracks val   :', split_tracks(args.data_root, 'val', val_tracks))
    print('backbone     :', args.backbone, '| dim', args.dim, '| size', args.size)
    print('out          :', out_path)

    train_loader = load_data(args.data_root, 'train', val_tracks=val_tracks, size=(args.size, args.size),
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             augment=not args.no_augment, limit=args.limit)
    val_loader = load_data(args.data_root, 'val', val_tracks=val_tracks, size=(args.size, args.size),
                           batch_size=args.batch_size, num_workers=args.num_workers,
                           augment=False, limit=args.limit)
    print('batches      : train=%d val=%d' % (len(train_loader), len(val_loader)))

    if args.no_weights:
        weights = None
        print('class_weights: DESACTIVADOS')
    else:
        ds = STKSegmentationDataset(args.data_root, split_tracks(args.data_root, 'train', val_tracks),
                                    size=(args.size, args.size))
        weights = compute_class_weights(ds, cache_path=args.weights,
                                        power=args.weight_power,
                                        clip=args.weight_clip).to(device)
        print('class_weights:', [round(float(v), 4) for v in weights])

    model = SAM2Segmenter(num_classes=NUM_CLASSES, backbone=args.backbone,
                          dim=args.dim, image_size=args.size,
                          freeze_encoder=True,
                          unfreeze_blocks=args.unfreeze_blocks,
                          unfreeze_neck=args.unfreeze_neck,
                          device=str(device)).to(device)
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    congelados = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    enc_params = model.encoder_trainable_parameters()
    print('parametros   : %.2f M entrenables / %.2f M congelados'
          % (entrenables / 1e6, congelados / 1e6))
    if enc_params:
        print('descongelado : %d bloques finales%s -> %.2f M con lr %.1e'
              % (args.unfreeze_blocks, ' + cuello FPN' if args.unfreeze_neck else '',
                 sum(p.numel() for p in enc_params) / 1e6, args.encoder_lr))
    else:
        print('descongelado : ninguno, encoder totalmente congelado')

    criterion = nn.CrossEntropyLoss(weight=weights)
    grupos = [{'params': list(model.head.parameters()), 'lr': args.lr}]
    if enc_params:
        grupos.append({'params': enc_params, 'lr': args.encoder_lr})
    optimizer = torch.optim.AdamW(grupos, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4)

    start_epoch, best_miou = 0, -1.0
    best_epoch, best_iou, since_best = -1, None, 0
    historial = []
    if args.resume is not None:
        resume_path = args.resume if args.resume and os.path.isfile(args.resume) else last_ckpt
        if os.path.isfile(resume_path):
            ck = torch.load(resume_path, map_location='cpu', weights_only=False)
            model.head.load_state_dict(ck['head'])
            if ck.get('encoder'):
                _, sobran = model.encoder.load_state_dict(ck['encoder'], strict=False)
                if sobran:
                    raise RuntimeError('claves desconocidas en el encoder: %s' % sobran[:3])
            optimizer.load_state_dict(ck['optimizer'])
            if 'scheduler' in ck:
                scheduler.load_state_dict(ck['scheduler'])
            start_epoch = ck['epoch'] + 1
            best_miou = ck.get('best_miou', -1.0)
            print('resume desde %s (epoca %d, best_miou %.4f)'
                  % (resume_path, start_epoch, best_miou))
        else:
            print('--resume pedido pero no hay checkpoint en %s; empiezo de cero' % resume_path)

    last_cm = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), mask)
            loss.backward()
            optimizer.step()
            run_loss += float(loss.detach())
            nb += 1
        train_loss = run_loss / max(nb, 1)

        cm, val_loss = evaluate(model, val_loader, device, criterion)
        miou = cm.miou
        if miou == miou:
            historial.append(miou)
        scheduler.step(miou)
        lr_now = optimizer.param_groups[0]['lr']
        last_cm = cm

        print()
        print(format_iou_table(
            cm.class_iou, miou,
            extra={'acc_global': '%.4f' % cm.global_accuracy,
                   'loss_train': '%.4f' % train_loss,
                   'loss_val': '%.4f' % val_loss,
                   'lr': '%.1e' % lr_now,
                   'seg': '%.1f' % (time.time() - t0)},
            title='EPOCH %d | VAL' % epoch), flush=True)

        if miou == miou and miou > best_miou:
            best_miou, best_epoch = miou, epoch
            best_iou = [float(v) for v in cm.class_iou]
            since_best = 0
            save_model(model, out_path)
            print('nuevo mejor mIoU %.4f -> %s' % (best_miou, out_path))
        else:
            since_best += 1
            if args.early_stop and since_best >= args.early_stop:
                print('\nparada temprana: %d epocas sin mejorar (mejor: epoca %d, %.4f)'
                      % (since_best, best_epoch, best_miou))
                break

        entrenados = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
        torch.save({'head': model.head.state_dict(),
                    'encoder': {k: v for k, v in model.encoder.state_dict().items()
                                if k in entrenados},
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'epoch': epoch, 'best_miou': best_miou, 'args': vars(args)}, last_ckpt)
        _log_run(args, epoch, train_loss, miou, cm.class_iou, lr_now)

    if last_cm is not None:
        print()
        print(format_confusion(last_cm))
    top5 = float('nan')
    sigma = float('nan')
    if historial:
        mejores = sorted(historial)[-5:]
        top5 = sum(mejores) / len(mejores)
        cola = historial[-10:]
        if len(cola) > 1:
            media = sum(cola) / len(cola)
            sigma = (sum((x - media) ** 2 for x in cola) / len(cola)) ** 0.5
    print('\nFIN. pico %.4f en la epoca %d | top5 %.4f | sigma %.4f  (modelo en %s)'
          % (best_miou, best_epoch, top5, sigma, out_path))
    return {'best_miou': best_miou, 'best_epoch': best_epoch, 'top5': top5,
            'sigma': sigma, 'class_iou': best_iou or [float('nan')] * NUM_CLASSES,
            'out': out_path}


def _log_run(args, epoch, train_loss, miou, class_iou, lr):
    path = os.path.abspath(args.runs_csv)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['timestamp', 'backbone', 'size', 'dim', 'lr', 'weight_power',
                        'augment', 'epoch', 'train_loss', 'val_miou']
                       + ['iou_' + c for c in CLASS_NAMES] + ['notes'])
        w.writerow([datetime.now().isoformat(timespec='seconds'), args.backbone,
                    args.size, args.dim, '%.2e' % lr, args.weight_power,
                    int(not args.no_augment), epoch, '%.4f' % train_loss, '%.4f' % miou]
                   + ['%.4f' % float(v) if float(v) == float(v) else 'nan' for v in class_iou]
                   + [args.notes])


def main():
    ap = argparse.ArgumentParser(description='Entrena la cabeza sobre SAM 2')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--backbone', default='small',
                    choices=['tiny', 'small', 'base_plus', 'large'])
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default='model.th')
    ap.add_argument('--ckpt-dir', default='salidas/checkpoints')
    ap.add_argument('--resume', nargs='?', const='', default=None)
    ap.add_argument('--weights', default='class_weights.json')
    ap.add_argument('--no-weights', action='store_true')
    ap.add_argument('--weight-power', type=float, default=0.30)
    ap.add_argument('--weight-clip', type=float, default=10.0)
    ap.add_argument('--no-augment', action='store_true')
    ap.add_argument('--unfreeze-blocks', type=int, default=0,
                    help='numero de bloques finales del trunk a descongelar')
    ap.add_argument('--unfreeze-neck', action='store_true',
                    help='descongelar tambien el cuello FPN (0.4 M)')
    ap.add_argument('--encoder-lr', type=float, default=1e-4,
                    help='lr de las partes descongeladas del encoder')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--runs-csv', default='experimentos/resultados/runs.csv')
    ap.add_argument('--notes', default='')
    ap.add_argument('--early-stop', type=int, default=0)
    ap.add_argument('--val-tracks', nargs='+', default=None,
                    help='circuitos de validacion; por defecto utils.VAL_TRACKS')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
