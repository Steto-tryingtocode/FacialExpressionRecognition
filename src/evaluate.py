"""
Valutazione dei modelli allenati con src/train.py.
Copre: predizioni sul test set, confusion matrix, metriche per classe
(precision/recall/F1 - fondamentali con classi sbilanciate come 'disgust'),
curve di training/validation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score


@torch.no_grad()
def get_predictions(model, loader, device=None):
    """Esegue il modello su tutto il loader e ritorna label vere e predette."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def plot_training_curves(history, title='Training curves'):
    """Plotta loss e accuracy di train/val affiancate, dalla history
    ritornata da train_model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='train')
    axes[1].plot(history['val_acc'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, title='Confusion matrix'):
    """Confusion matrix come heatmap. normalize=True mostra le percentuali
    per riga (per classe vera), utile perché con classi sbilanciate i conteggi
    assoluti sono fuorvianti."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1 if normalize else None)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predetta')
    ax.set_ylabel('Vera')
    ax.set_title(title)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

    return cm


def per_class_report(y_true, y_pred, class_names):
    """Precision/recall/F1 per classe, non solo accuracy globale.
    Fondamentale per capire come si comporta il modello su 'disgust'."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Macro F1 (media semplice tra classi, penalizza le classi piccole se vanno male): {macro_f1:.3f}")
    print(f"Weighted F1 (pesata per frequenza di classe): {weighted_f1:.3f}")

    return {'macro_f1': macro_f1, 'weighted_f1': weighted_f1}


def evaluate_model(model, test_loader, class_names, history=None, device=None):
    """Pipeline completa di valutazione: curve (se history fornita),
    predizioni sul test, confusion matrix, report per classe.
    Ritorna y_true, y_pred per eventuali analisi aggiuntive."""
    if history is not None:
        plot_training_curves(history)

    y_true, y_pred = get_predictions(model, test_loader, device)

    plot_confusion_matrix(y_true, y_pred, class_names)
    per_class_report(y_true, y_pred, class_names)

    return y_true, y_pred