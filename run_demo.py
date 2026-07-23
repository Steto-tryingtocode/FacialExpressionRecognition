import sys
sys.path.append('.')

from src.models import DeepCNN
from src.data import load_arrays
from src.inference import run_webcam_demo
import torch

# stesso mapping usato in training — se non lo hai già in mano, ricavalo da un CSV:
# import pandas as pd
# df = pd.read_csv('dataset/processed/train_final_index.csv')
# label2idx = {e: i for i, e in enumerate(sorted(df['emotion'].unique()))}
label2idx = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
class_names = sorted(label2idx, key=label2idx.get)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DeepCNN(num_classes=7).to(device)
model.load_state_dict(torch.load('results/deep_cnn_focal_lightweight.pt', map_location=device))

data = load_arrays('dataset/processed/fer_arrays.npz')

run_webcam_demo(model, data['mean'], data['std'], class_names, device)