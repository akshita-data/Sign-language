import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset Folder
DATASET_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

print("Project Root :", PROJECT_ROOT)
print("Dataset Path :", DATASET_PATH)

if DATASET_PATH.exists():
    print("✅ Dataset folder found.")
else:
    print("❌ Dataset folder not found.")

folders = [
    folder
    for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
]

print(f"Found {len(folders)} sign folders.")

print("\n========== DATASET VALIDATION REPORT ==========\n")

print(f"Dataset Folder : FOUND")
print(f"Number of Sign Folders : {len(folders)}\n")

for i, folder in enumerate(sorted(folders), start=1):
    print(f"{i}. {folder.name}")

print("\n==============================================")