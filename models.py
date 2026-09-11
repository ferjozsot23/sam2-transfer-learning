"""Segmentacion semantica multiclase por transfer learning desde SAM 2.

SAM 2 no es un segmentador semantico: es promptable y binario. Aqui se usa solo
su image encoder (Hiera + cuello FPN) como extractor de features, se descartan la
memory attention y el mask decoder original, y se anade una cabeza nueva que
fusiona los tres niveles del FPN y produce 7 logits por pixel. El encoder puede
quedar congelado o descongelarse por bloques; el modelo publicado lo afina entero.

El encoder devuelve backbone_fpn con 3 niveles de 256 canales a strides 4, 8 y 16,
y acepta entradas multiplo de 64. El forward reescala internamente al tamano de
trabajo y devuelve los logits al tamano original de la imagen.

Este archivo esta separado del entrenamiento a proposito: permite hacer
`from models import load_model` sin disparar nada mas.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 7

BACKBONES = {
    'tiny': 'facebook/sam2.1-hiera-tiny',
    'small': 'facebook/sam2.1-hiera-small',
    'base_plus': 'facebook/sam2.1-hiera-base-plus',
    'large': 'facebook/sam2.1-hiera-large',
}

SAM2_MEAN = (0.485, 0.456, 0.406)
SAM2_STD = (0.229, 0.224, 0.225)


def build_encoder(backbone='small', device='cpu', pretrained=True):
    os.environ.setdefault('SAM2_BUILD_CUDA', '0')
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2, build_sam2_hf

    if backbone not in BACKBONES:
        raise ValueError('backbone %r no reconocido; usa uno de %s'
                         % (backbone, list(BACKBONES)))
    if pretrained:
        sam = build_sam2_hf(BACKBONES[backbone], device=device)
    else:
        config, _ = HF_MODEL_ID_TO_FILENAMES[BACKBONES[backbone]]
        sam = build_sam2(config, ckpt_path=None, device=device)
    encoder = sam.image_encoder
    del sam
    return encoder


class FPNHead(nn.Module):
    def __init__(self, in_channels=256, dim=128, num_classes=NUM_CLASSES, levels=3):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(in_channels, dim, 1) for _ in range(levels))
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                          nn.BatchNorm2d(dim),
                          nn.ReLU(inplace=True))
            for _ in range(levels))
        self.classifier = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_classes, 1))

    def forward(self, feats):
        x = self.smooth[-1](self.lateral[-1](feats[-1]))
        for i in range(len(feats) - 2, -1, -1):
            lat = self.lateral[i](feats[i])
            x = F.interpolate(x, size=lat.shape[-2:], mode='bilinear', align_corners=False)
            x = self.smooth[i](lat + x)
        return self.classifier(x)


class SAM2Segmenter(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, backbone='small', dim=128,
                 image_size=448, freeze_encoder=True, unfreeze_blocks=0,
                 unfreeze_neck=False, pretrained=True, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.dim = dim
        self.image_size = int(image_size)
        self.freeze_encoder = freeze_encoder
        self.unfreeze_blocks = int(unfreeze_blocks)
        self.unfreeze_neck = bool(unfreeze_neck)

        self.register_buffer('mean', torch.tensor(SAM2_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(SAM2_STD).view(1, 3, 1, 1))

        self.encoder = build_encoder(backbone, device=device, pretrained=pretrained)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            if self.unfreeze_blocks > 0:
                for b in self.encoder.trunk.blocks[-self.unfreeze_blocks:]:
                    b.requires_grad_(True)
            if self.unfreeze_neck:
                self.encoder.neck.requires_grad_(True)
        self.encoder_congelado = not any(p.requires_grad
                                         for p in self.encoder.parameters())

        with torch.no_grad():
            probe = self.encoder(torch.zeros(1, 3, 64, 64, device=device))['backbone_fpn']
        self.head = FPNHead(in_channels=probe[0].shape[1], dim=dim,
                            num_classes=num_classes, levels=len(probe))

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_congelado:
            self.encoder.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encoder_trainable_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def forward(self, x):
        h, w = x.shape[-2:]
        x = (x - self.mean) / self.std
        if (h, w) != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)
        if self.encoder_congelado:
            with torch.no_grad():
                feats = self.encoder(x)['backbone_fpn']
            feats = [f.detach() for f in feats]
        else:
            feats = self.encoder(x)['backbone_fpn']
        logits = self.head(feats)
        return F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)


def save_model(model, path='model.th'):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    limpio = lambda sd: {k.replace('module.', ''): v.detach().cpu() for k, v in sd.items()}
    estado = {
        'head': limpio(model.head.state_dict()),
        'meta': {'num_classes': model.num_classes, 'backbone': model.backbone,
                 'dim': model.dim, 'image_size': model.image_size,
                 'freeze_encoder': model.freeze_encoder,
                 'unfreeze_blocks': model.unfreeze_blocks,
                 'unfreeze_neck': model.unfreeze_neck},
    }
    if not model.encoder_congelado:
        entrenables = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.encoder.parameters())
        if entrenables / total > 0.5:
            estado['encoder'] = limpio(model.encoder.state_dict())
        else:
            entrenados = {n for n, p in model.encoder.named_parameters() if p.requires_grad}
            estado['encoder_parcial'] = {k: v for k, v in limpio(model.encoder.state_dict()).items()
                                         if k in entrenados}
    torch.save(estado, path)
    return path


def load_model(path='model.th', device='cpu'):
    estado = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    meta = estado['meta']
    completo = 'encoder' in estado
    model = SAM2Segmenter(num_classes=meta['num_classes'], backbone=meta['backbone'],
                          dim=meta['dim'], image_size=meta['image_size'],
                          freeze_encoder=meta['freeze_encoder'],
                          unfreeze_blocks=meta.get('unfreeze_blocks', 0),
                          unfreeze_neck=meta.get('unfreeze_neck', False),
                          pretrained=not completo, device='cpu')
    model.head.load_state_dict(estado['head'])
    if completo:
        model.encoder.load_state_dict(estado['encoder'])
    elif 'encoder_parcial' in estado:
        _, sobran = model.encoder.load_state_dict(estado['encoder_parcial'], strict=False)
        if sobran:
            raise RuntimeError('claves desconocidas en el encoder: %s' % sobran[:3])
    return model.to(device).eval()


if __name__ == '__main__':
    import tempfile

    m = SAM2Segmenter()
    m.eval()

    entrenables = sum(p.numel() for p in m.parameters() if p.requires_grad)
    congelados = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print('parametros entrenables: %9d  (%.2f M)' % (entrenables, entrenables / 1e6))
    print('parametros congelados : %9d  (%.2f M)' % (congelados, congelados / 1e6))
    assert entrenables > 0, 'la cabeza esta congelada'
    assert congelados > entrenables, 'el encoder no esta congelado'
    cabeza = sum(p.numel() for p in m.head.parameters())
    assert cabeza == entrenables, 'hay parametros entrenables fuera de la cabeza'

    for hw in [(448, 448), (400, 400), (256, 320), (211, 333), (480, 640)]:
        with torch.no_grad():
            y = m(torch.rand(1, 3, *hw))
        assert y.shape == (1, NUM_CLASSES, *hw), (hw, tuple(y.shape))
        print('in %-12s -> out %s  OK' % (str(hw), tuple(y.shape)))

    with tempfile.TemporaryDirectory() as tmp:
        ruta = os.path.join(tmp, 'prueba.th')
        save_model(m, ruta)
        print('tamano del .th: %.2f MB' % (os.path.getsize(ruta) / 1e6))
        m2 = load_model(ruta)
        with torch.no_grad():
            x = torch.rand(1, 3, 400, 400)
            assert torch.allclose(m(x), m2(x), atol=1e-5)
    print('save_model/load_model round-trip OK')
