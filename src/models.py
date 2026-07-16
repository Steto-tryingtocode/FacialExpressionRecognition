"""
Architetture per la classificazione delle espressioni facciali su FER2013.
Input atteso: tensori (batch, 1, 48, 48), già normalizzati (vedi src/data.py).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class BaselineCNN(nn.Module):
    """CNN piccola (3 blocchi conv), da usare come punto di partenza per
    dimostrare il valore dei layer convoluzionali rispetto a un modello lineare."""

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48 -> 24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24 -> 12

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class DeepCNN(nn.Module):
    """CNN più profonda con batch norm, blocchi doppi per stadio.
    Confronto per valutare il guadagno di capacità/regolarizzazione
    rispetto alla baseline."""

    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 64),     # 48 -> 24
            conv_block(64, 128),   # 24 -> 12
            conv_block(128, 256),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18_transfer(num_classes=7, freeze_backbone=True):
    """ResNet18 pretrainata su ImageNet, adattata a input 1 canale.
    Utile come confronto rispetto alle CNN allenate da zero.

    Nota: ImageNet è RGB 224x224, quindi in src/data.py serve un resize
    a 224x224 e replicare il canale grayscale a 3 canali prima di passare
    le immagini a questo modello (non serve invece per BaselineCNN/DeepCNN,
    che lavorano nativamente a 48x48x1)."""
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

    # adatta il primo layer conv a input 1 canale, mantenendo i pesi
    # pretrainati mediati sui 3 canali originali
    old_conv = model.conv1
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                           stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.conv1 = new_conv

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # sanity check rapido sulle shape
    x = torch.randn(4, 1, 48, 48)

    for name, model in [('BaselineCNN', BaselineCNN()), ('DeepCNN', DeepCNN())]:
        out = model(x)
        print(f"{name}: output {out.shape}, parametri allenabili {count_parameters(model):,}")