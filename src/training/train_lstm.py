from pathlib import Path
import json
import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import (
    Input,
    Masking,
    LSTM,
    Dense,
    Dropout
)
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_PATH = PROJECT_ROOT / "dataset" / "processed_landmarks"

MODEL_PATH = PROJECT_ROOT / "models"
MODEL_PATH.mkdir(exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

X = np.load(PROCESSED_PATH / "X.npy")
y = np.load(PROCESSED_PATH / "y.npy")

with open(PROCESSED_PATH / "label_map.json") as f:
    label_map = json.load(f)

print("Dataset loaded successfully.\n")

print("X:", X.shape)
print("y:", y.shape)

print(label_map)

# --------------------------------------------------
# Split Dataset
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

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# --------------------------------------------------
# Labels
# --------------------------------------------------

NUM_CLASSES = len(label_map)

y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = Sequential()

model.add(Input(shape=(60, 63)))

# Ignore padded frames
model.add(Masking(mask_value=0.0))

model.add(
    LSTM(
        64,
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        32
    )
)

model.add(Dense(32, activation="relu"))

model.add(Dropout(0.3))

model.add(
    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
)

model.summary()

# --------------------------------------------------
# Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=15,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH / "best_model.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# --------------------------------------------------
# Train
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ],
    verbose=1
)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")