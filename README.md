# Facial Expression Recognition — FER2013

Progetto per il corso di Numerical Analysis for Machine Learning (NAML).
Classificazione dell'espressione facciale su 7 classi (angry, disgust, fear,
happy, neutral, sad, surprise) usando il dataset [FER2013](https://www.kaggle.com/datasets/msambare/fer2013),
con un'estensione applicativa di riconoscimento in tempo reale da webcam.

## Struttura del progetto

```
FacialExpressionRecognition/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA: distribuzione classi, duplicati, sample
│   ├── 03_experiments.ipynb      # training e valutazione dei modelli
│   └── 04_error_analysis.ipynb   # analisi qualitativa degli errori
├── src/
│   ├── data.py                   # caricamento dati, normalizzazione, Dataset/DataLoader
│   ├── models.py                 # BaselineCNN, DeepCNN, ResNet18 transfer
│   ├── train.py                  # training loop, class weighting, focal loss, scheduler
│   ├── evaluate.py                # confusion matrix, metriche per classe, curve
│   └── inference.py              # demo webcam real-time
├── dataset/
│   ├── raw/                      # immagini originali (train/test per classe)
│   └── processed/                # indici puliti (CSV) e array precomputati (.npz)
├── results/                      # checkpoint dei modelli allenati (.pt)
├── run_demo.py                   # script di avvio della demo webcam
└── requirements.txt
```

Nota: `dataset/raw`, `dataset/processed` e `results/` non sono versionati
(file grossi/binari) — vanno rigenerati eseguendo i notebook in ordine, o
copiati manualmente se si lavora su più macchine (vedi sezione Windows/WSL).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Linux/WSL
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**PyTorch con supporto GPU (CUDA)**: `requirements.txt` non fissa la build
di `torch`/`torchvision`, perché dipende dalla piattaforma. Se hai una GPU
NVIDIA, installa la build CUDA corretta *dopo* il requirements generico:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(la versione `cuXXX` dipende dal driver — verifica con `nvidia-smi` quale
versione CUDA supporta; le build sono retrocompatibili con driver più recenti)

Verifica che la GPU sia effettivamente rilevata prima di lanciare un training lungo:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset

Scarica [FER2013 da Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
ed estrai in `dataset/raw/`, mantenendo la struttura `train/<classe>/*.jpg`
e `test/<classe>/*.jpg`.

## Ordine di esecuzione

1. **`01_eda.ipynb`** — esplora il dataset, rimuove i duplicati (interni a
   ciascuno split e tra train/test, per evitare data leakage), produce lo
   split train/val stratificato, salva gli indici puliti in `dataset/processed/`.

2. **`03_experiments.ipynb`** — costruisce gli array precomputati
   (`build_and_save_arrays`, da eseguire una sola volta o quando cambiano
   gli indici), allena i modelli e valuta i risultati. Contiene i vari
   esperimenti (architettura, loss, class weighting) descritti sotto.

3. **`04_error_analysis.ipynb`** — analisi qualitativa degli errori sul
   modello migliore: quali classi si confondono di più, esempi visivi,
   confidenza del modello sulle predizioni sbagliate.

## Modelli e risultati

Tre architetture confrontate (`src/models.py`):

- **`BaselineCNN`** — CNN piccola, 3 blocchi convoluzionali, ~680k parametri
- **`DeepCNN`** — più profonda, batch norm, ~3.5M parametri
- **`build_resnet18_transfer`** — ResNet18 pretrainata su ImageNet, adattata
  a input grayscale 1 canale (resize a 224×224 richiesto, vedi
  `get_dataloaders(..., resize_for_resnet=True)`)

Strategie di bilanciamento per la classe minoritaria (`disgust`, ~1.5% del
dataset), configurabili in `train_model`: class weighting (`power`
regolabile), focal loss, o combinazione delle due.

| Modello | Loss | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| BaselineCNN | CE pesata | 59.3% | 0.549 | 0.590 |
| DeepCNN | CE pesata | 62.7% | 0.582 | 0.629 |
| DeepCNN | Focal + class weights pieni | 53.1% | 0.491 | 0.544 |
| DeepCNN | Focal, senza class weights | 65.1% | 0.592 | 0.644 |
| **DeepCNN** | **Focal + class weights leggeri (power=0.3) + weight_decay=5e-4** | **64.9%** | **0.618** | **0.648** |
| ResNet18 (backbone congelato) | CE pesata | 41.9% | 0.345 | 0.404 |

Il modello migliore (`results/deep_cnn_focal_lightweight.pt`) è quello usato
di default nella demo webcam. L'accuracy umana media riportata in letteratura
su FER2013 è circa 65-68%, quindi i risultati sono nell'intorno delle
prestazioni umane su un dataset intrinsecamente ambiguo/rumoroso.

## Demo webcam

`src/inference.py` implementa il riconoscimento in tempo reale: rileva volti
con Haar Cascade (OpenCV) e classifica l'espressione con il modello allenato,
con smoothing delle predizioni su una finestra mobile di frame.

```bash
python run_demo.py
```
Premi `q` per uscire.

**Nota Windows/WSL**: se sviluppi in WSL, l'accesso diretto alla webcam
può non funzionare senza configurazione aggiuntiva (`usbipd-win`). In
quel caso, esegui `run_demo.py` da un venv Windows nativo separato,
copiando i file necessari dal lato WSL:
- `results/<checkpoint>.pt` (modello allenato)
- `dataset/processed/fer_arrays.npz` (statistiche di normalizzazione)

Da WSL, copia con:
```bash
cp results/deep_cnn_focal_lightweight.pt /mnt/c/percorso/alla/repo/results/
cp dataset/processed/fer_arrays.npz /mnt/c/percorso/alla/repo/dataset/processed/
```

**Limite noto**: il modello è allenato su FER2013 (immagini già ritagliate,
scala di grigi, dataset controllato). Le performance "dal vivo" su webcam
sono visibilmente inferiori a quelle sul test set, per il gap di dominio
(illuminazione, angolazione, qualità del ritaglio del volto rispetto al
dataset di training).

## Limiti noti del progetto

- **FER2013 è un dataset rumoroso**: alcune immagini sono mal etichettate,
  contengono watermark, o non sono facce — limite intrinseco discusso in
  `01_eda.ipynb`.
- **`disgust`** resta la classe più difficile per lo sbilanciamento estremo
  (~1.5% dei dati); nessuna strategia di bilanciamento provata elimina del
  tutto il trade-off precision/recall su questa classe.
- **`fear`** si confonde sistematicamente con `sad` e `surprise` in tutti
  gli esperimenti — vedi `04_error_analysis.ipynb` per l'analisi qualitativa.
- **Transfer learning (ResNet18)** con backbone congelato ha performato
  nettamente peggio delle CNN allenate da zero, probabilmente per il forte
  gap di dominio tra ImageNet (foto naturali RGB) e FER2013 (volti grayscale
  a bassa risoluzione nativa).