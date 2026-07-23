"""
Demo real-time: rileva volti da webcam (Haar Cascade OpenCV) e classifica
l'espressione facciale con un modello allenato (src/models.py).

Da eseguire come script standalone (non da notebook Jupyter, cv2.imshow
non si comporta bene dentro Jupyter). Pensato per girare su Windows nativo
se la webcam non e' accessibile da WSL.
"""

import collections
import cv2
import numpy as np
import torch

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def preprocess_face(face_bgr, mean, std, size=48):
    """Converte un ritaglio di volto (BGR, da OpenCV) nel formato atteso
    dal modello: (1, 1, size, size), grayscale, normalizzato con le stesse
    statistiche usate in training."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(resized.astype(np.float32)) / 255.0
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)


@torch.no_grad()
def predict_emotion(model, face_bgr, mean, std, class_names, device, size=48):
    """Ritorna (etichetta_predetta, vettore_probabilita) per un singolo volto."""
    input_tensor = preprocess_face(face_bgr, mean, std, size).to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return class_names[pred_idx], probs


class PredictionSmoother:
    """Media mobile sulle ultime N predizioni per un volto, per evitare che
    l'etichetta mostrata 'sfarfalli' da un frame all'altro. Tiene una
    finestra separata per posizione approssimativa del volto (utile con
    piu' persone in campo)."""

    def __init__(self, window=15):
        self.window = window
        self.history = collections.deque(maxlen=window)

    def update(self, probs):
        self.history.append(probs)
        avg_probs = np.mean(self.history, axis=0)
        return avg_probs


def run_webcam_demo(model, mean, std, class_names, device,
                     camera_index=0, min_face_size=(60, 60), smooth_window=15):
    """Loop principale: cattura frame, rileva volti, classifica, disegna
    box + etichetta. Premi 'q' per uscire."""
    model.eval()
    model = model.to(device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire la webcam (camera_index={camera_index})")

    smoother = PredictionSmoother(window=smooth_window)

    print("Demo avviata. Premi 'q' nella finestra video per uscire.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame non letto, interrompo.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size
        )

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]

            _, probs = predict_emotion(model, face_crop, mean, std, class_names, device)
            smoothed_probs = smoother.update(probs)
            pred_idx = int(smoothed_probs.argmax())
            emotion = class_names[pred_idx]
            confidence = smoothed_probs[pred_idx]

            label = f"{emotion} ({confidence:.0%})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Facial Expression Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()