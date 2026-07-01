from pathlib import Path

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_VIDEO_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

OUTPUT_PATH = PROJECT_ROOT / "dataset" / "processed_landmarks"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Start
# ---------------------------------------------------

print("\n========== VIDEO LANDMARK EXTRACTOR ==========\n")

if not RAW_VIDEO_PATH.exists():
    print("❌ ERROR : Raw video folder not found.")
    exit()

print("✅ Raw video folder found.")

sign_folders = sorted(
    [folder for folder in RAW_VIDEO_PATH.iterdir() if folder.is_dir()],
    key=lambda folder: folder.name
)

print(f"\nFound {len(sign_folders)} sign folders.")

all_videos = []

for folder in sign_folders:

    videos = sorted(folder.glob("*.mp4"))

    for video in videos:
        all_videos.append(video)

print(f"Total Videos Found : {len(all_videos)}")
print(f"Output Folder      : {OUTPUT_PATH}")