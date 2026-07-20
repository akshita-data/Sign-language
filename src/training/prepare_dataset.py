from pathlib import Path
import pandas as pd
import numpy as np
import json

SEQUENCE_LENGTH = 60

def pad_or_truncate(sequence):

    num_frames = sequence.shape[0]

    if num_frames > SEQUENCE_LENGTH:
        return sequence[:SEQUENCE_LENGTH]

    if num_frames < SEQUENCE_LENGTH:

        padding = np.zeros(
            (SEQUENCE_LENGTH - num_frames, sequence.shape[1])
        )

        return np.vstack((sequence, padding))

    return sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANDMARK_PATH = PROJECT_ROOT / "dataset" / "landmarks"
PROCESSED_PATH = PROJECT_ROOT / "dataset" / "processed_landmarks"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

if not LANDMARK_PATH.exists():
    raise FileNotFoundError("Landmarks directory not found.")

print("Landmarks directory found successfully.")

sign_folders = sorted(
    folder for folder in LANDMARK_PATH.iterdir()
    if folder.is_dir()
)

label_map = {
    folder.name: index
    for index, folder in enumerate(sign_folders)
}

print("\nLabel Map:")
print(label_map)

print(f"\nFound {len(sign_folders)} sign folders.")

label_map = {
    folder.name: index
    for index, folder in enumerate(sign_folders)
}

print("\nLabel Map:")
print(label_map)

X = []
y = []

sequence_lengths = []

for sign_folder in sign_folders:

    print(f"\nSign: {sign_folder.name}")

    csv_files = sorted(sign_folder.glob("*.csv"))

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)
        sequence_lengths.append(len(df))

        sequence = pad_or_truncate(df.values)

        X.append(sequence)
        y.append(label_map[sign_folder.name])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

np.save(PROCESSED_PATH / "X.npy", X)
np.save(PROCESSED_PATH / "y.npy", y)

with open(PROCESSED_PATH / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("\nProcessed dataset saved successfully.")

print(f"\nDataset shape : {X.shape}")
print(f"Labels shape  : {y.shape}")

print(f"\nFirst sample shape : {X[0].shape}")
print(f"First sample label : {y[0]}")