"""
Caricamento dati, normalizzazione e Dataset/DataLoader per FER2013.
Riusa gli indici puliti prodotti in 01_eda.ipynb e gli array precomputati
prodotti in 02_preprocessing.ipynb.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROCESSED_DIR = os.path.join('..', 'dataset', 'processed')


def normalize_path(path_str):
    """Converte un path in forma portabile, indipendentemente da quale OS
    l'ha scritto nel CSV (Windows salva con backslash, che su Linux/WSL
    non viene interpretato come separatore)."""
    return str(Path(path_str.replace('\\', '/')))


def load_label_map(df):
    """Costruisce il mapping emotion -> indice intero, ordinato alfabeticamente
    per garantire lo stesso mapping in ogni notebook/run."""
    emotions = sorted(df['emotion'].unique())
    return {emotion: idx for idx, emotion in enumerate(emotions)}


def load_images_to_array(df, label2idx, size=48):
    """Carica tutte le immagini indicate nel DataFrame in un array numpy uint8."""
    arr = np.zeros((len(df), size, size), dtype=np.uint8)
    for i, path in enumerate(df['path']):
        arr[i] = np.array(Image.open(normalize_path(path)))
    labels = df['emotion'].map(label2idx).values.astype(np.int64)
    return arr, labels


def compute_normalization_stats(X_train):
    """Media e std calcolate SOLO sul train, da riusare identiche su val/test."""
    mean = X_train.mean() / 255.0
    std = X_train.std() / 255.0
    return mean, std


def build_and_save_arrays(train_csv, val_csv, test_csv, out_path, size=48):
    """Pipeline completa: dai CSV puliti agli array precomputati salvati su disco.
    Va eseguita una volta sola (o quando cambiano gli indici puliti)."""
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    label2idx = load_label_map(df_train)

    X_train, y_train = load_images_to_array(df_train, label2idx, size)
    X_val, y_val = load_images_to_array(df_val, label2idx, size)
    X_test, y_test = load_images_to_array(df_test, label2idx, size)

    mean, std = compute_normalization_stats(X_train)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
              X_train=X_train, y_train=y_train,
              X_val=X_val, y_val=y_val,
              X_test=X_test, y_test=y_test,
              mean=mean, std=std)

    return label2idx


def load_arrays(npz_path):
    """Ricarica gli array precomputati salvati da build_and_save_arrays."""
    data = np.load(npz_path)
    return {
        'X_train': data['X_train'], 'y_train': data['y_train'],
        'X_val': data['X_val'], 'y_val': data['y_val'],
        'X_test': data['X_test'], 'y_test': data['y_test'],
        'mean': float(data['mean']), 'std': float(data['std']),
    }


class FERDataset(Dataset):
    """Dataset PyTorch per FER2013. Applica normalizzazione (sempre) e
    augmentation (solo se train=True), più aggressiva sulla classe minoritaria
    indicata da disgust_idx."""

    def __init__(self, images, labels, mean, std, train=False, disgust_idx=None,
                 resize_for_resnet=False):
        self.images = images
        self.labels = labels
        self.mean = mean
        self.std = std
        self.train = train
        self.disgust_idx = disgust_idx
        self.resize_for_resnet = resize_for_resnet

        self.base_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])

        self.strong_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.train:
            if self.disgust_idx is not None and label == self.disgust_idx:
                img = self.strong_augment(img)
            else:
                img = self.base_augment(img)

        img_t = torch.from_numpy(np.array(img, dtype=np.float32)) / 255.0
        img_t = (img_t - self.mean) / self.std
        img_t = img_t.unsqueeze(0)  # (1, 48, 48)

        if self.resize_for_resnet:
            img_t = img_t.unsqueeze(0)  # (1, 1, 48, 48), serve la batch dim per interpolate
            img_t = torch.nn.functional.interpolate(img_t, size=(224, 224), mode='bilinear', align_corners=False)
            img_t = img_t.squeeze(0)  # (1, 224, 224) - Manteniamo un singolo canale

        return img_t, torch.tensor(label, dtype=torch.long)


def get_dataloaders(npz_path, label2idx, batch_size=64, num_workers=0, resize_for_resnet=False):
    """Costruisce train/val/test DataLoader a partire dagli array precomputati.
    num_workers=0 di default: su Windows/VS Code il multiprocessing dei
    DataLoader può bloccarsi, e con dataset così piccoli (tutto in RAM) non
    serve comunque.
    
    resize_for_resnet=True: produce tensori (1, 224, 224) invece di (1, 48, 48),
    necessario per build_resnet18_transfer (che è stata adattata per input a 
    singolo canale).
    Riduci batch_size (es. 32) se usi questa opzione, dato l'ingombro maggiore
    in VRAM rispetto alle immagini 48x48.
    """
    data = load_arrays(npz_path)
    disgust_idx = label2idx.get('disgust')

    train_dataset = FERDataset(data['X_train'], data['y_train'], data['mean'], data['std'],
                                 train=True, disgust_idx=disgust_idx, resize_for_resnet=resize_for_resnet)
    val_dataset = FERDataset(data['X_val'], data['y_val'], data['mean'], data['std'],
                               train=False, resize_for_resnet=resize_for_resnet)
    test_dataset = FERDataset(data['X_test'], data['y_test'], data['mean'], data['std'],
                                train=False, resize_for_resnet=resize_for_resnet)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader