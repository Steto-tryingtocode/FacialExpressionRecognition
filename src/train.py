"""
Training loop per i modelli di src/models.py.
Gestisce class weights (per lo sbilanciamento di 'disgust'), validazione
ad ogni epoca, early stopping e salvataggio del checkpoint migliore.
"""

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Alternativa a CrossEntropy: schiaccia il contributo alla loss dei
    campioni già classificati con sicurezza, concentrando il training sui
    casi difficili. gamma=0 equivale a CrossEntropy; gamma piu' alto
    enfatizza maggiormente i casi difficili."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def compute_class_weights(y_train, num_classes, power=0.3):
    """Pesi inversamente proporzionali alla frequenza di classe, normalizzati
    in modo che la media dei pesi sia 1 (mantiene la scala della loss
    confrontabile tra esperimenti con/senza weighting)."""
    counts = np.bincount(y_train, minlength=num_classes)
    weights = 1.0 / (counts ** power)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    """Esegue un'epoca di train o di validazione. Ritorna loss media e accuracy.
    Se scaler è fornito (solo su GPU), usa mixed precision per velocizzare."""
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, num_classes,
                 y_train=None, use_class_weights=True,
                 loss_type='ce', focal_gamma=2.0, power=0.3,
                 epochs=50, lr=1e-3, weight_decay=1e-4,
                 patience=7, device=None, checkpoint_path='best_model.pt',
                 use_scheduler=True, scheduler_factor=0.5, scheduler_patience=3,
                 verbose=True):
    """Training loop completo con early stopping su val loss.

    Parametri chiave:
    - y_train: array delle label del train, richiesto se use_class_weights=True
    - loss_type: 'ce' (CrossEntropy, default) oppure 'focal' (FocalLoss,
      utile quando class weighting da solo non basta per una classe difficile)
    - focal_gamma: usato solo se loss_type='focal'; piu' alto = piu' focus
      sui casi difficili (default 2.0, valore comune in letteratura)
    - patience: numero di epoche senza miglioramento sulla val loss prima di fermarsi
    - checkpoint_path: dove salvare i pesi del modello con val loss migliore
    - use_scheduler: se True, dimezza (di default) il LR quando la val loss
      si stabilizza per scheduler_patience epoche, invece di restare fisso

    Ritorna il modello con i pesi migliori caricati, e la history (dict di liste,
    include anche 'lr' per vedere quando lo scheduler è intervenuto)
    per plottare le curve dopo."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # velocizza se le shape di input sono fisse

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    if use_class_weights:
        if y_train is None:
            raise ValueError("y_train è richiesto quando use_class_weights=True")
        class_weights = compute_class_weights(y_train, num_classes).to(device)
    else:
        class_weights = None

    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focal':
        criterion = FocalLoss(weight=class_weights, gamma=focal_gamma)
    else:
        raise ValueError(f"loss_type deve essere 'ce' o 'focal', ricevuto '{loss_type}'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)

    # mixed precision: solo su GPU, non ha senso/non è supportato su CPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=None)

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if verbose:
            elapsed = time.time() - start
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={current_lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping all'epoca {epoch} "
                          f"(nessun miglioramento da {patience} epoche)")
                break

    model.load_state_dict(best_state)
    return model, history