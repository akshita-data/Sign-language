from pathlib import Path
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from keras.models import load_model
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_PATH = PROJECT_ROOT / "dataset" / "processed_landmarks"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

X = np.load(PROCESSED_PATH / "X.npy")
y = np.load(PROCESSED_PATH / "y.npy")

with open(PROCESSED_PATH / "label_map.json") as f:
    label_map = json.load(f)

class_names = list(label_map.keys())

# --------------------------------------------------
# Same train/validation/test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = load_model(MODEL_PATH)

# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45
)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()