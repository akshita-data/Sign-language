from pathlib import Path
import json
import os

import cv2
import numpy as np

from tensorflow.keras.models import load_model

from src.detection.hand_detection import (
    detect_hands,
    draw_hands,
    close_detector,
)

from src.detection.landmark_extraction import extract_landmarks

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

LABEL_MAP_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed_landmarks"
    / "label_map.json"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------
# Load Label Map
# --------------------------------------------------

print("Loading label map...")

with open(LABEL_MAP_PATH, "r") as file:
    label_map = json.load(file)
index_to_label = {v: k for k, v in label_map.items()}
print("Label map loaded successfully.")
print(label_map)


# --------------------------------------------------
# Webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")

# --------------------------------------------------
# Sequence Buffer
# --------------------------------------------------

sequence = []
SEQUENCE_LENGTH = 60

# --------------------------------------------------
# Webcam Loop
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detect_hands(rgb_frame)

    landmarks = extract_landmarks(results)

    if landmarks is not None:

        sequence.append(landmarks)
        

        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

    if len(sequence) == SEQUENCE_LENGTH:
        
        input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)

        prediction = model.predict(input_data, verbose=0)

        predicted_index = np.argmax(prediction)

        confidence = prediction[0][predicted_index]

        predicted_label = index_to_label[predicted_index]

        cv2.putText(
            frame,
            f"{predicted_label} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        print(f"{predicted_label} ({confidence:.2f})")

    draw_hands(frame, results)


    cv2.imshow("Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
close_detector()
cv2.destroyAllWindows()