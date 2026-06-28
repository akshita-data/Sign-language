from pathlib import Path
import csv

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

OUTPUT_CSV = PROJECT_ROOT / "dataset" / "metadata" / "dataset_inventory.csv"

# ---------------------------------------------------
# Check Dataset Folder
# ---------------------------------------------------

print("\n========== DATASET INVENTORY ==========\n")

print("Project Root :", PROJECT_ROOT)
print("Dataset Path :", DATASET_PATH)

if not DATASET_PATH.exists():
    print("\n❌ ERROR: Dataset folder not found.")
    exit()

# ---------------------------------------------------
# Find Sign Folders
# ---------------------------------------------------

sign_folders = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()],
    key=lambda x: x.name
)

print(f"\nFound {len(sign_folders)} sign folders.\n")

# ---------------------------------------------------
# Count Videos
# ---------------------------------------------------

inventory = []

for sign_folder in sign_folders:

    video_count = len(list(sign_folder.glob("*.mp4")))

    inventory.append([sign_folder.name, video_count])

    print(f"{sign_folder.name:<15} : {video_count} videos")

# ---------------------------------------------------
# Save CSV
# ---------------------------------------------------

with open(OUTPUT_CSV, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["SIGN_NAME", "VIDEO_COUNT"])

    writer.writerows(inventory)

print("\n✅ Dataset inventory saved successfully.")

print(f"Location : {OUTPUT_CSV}")