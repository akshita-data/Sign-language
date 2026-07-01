from pathlib import Path

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

# ---------------------------------------------------
# Validator
# ---------------------------------------------------

print("\n========== DATASET VALIDATOR ==========\n")

# Check dataset folder
if not DATASET_PATH.exists():
    print("❌ ERROR : Dataset folder not found.")
    exit()

print("✅ Dataset folder found.\n")

# Get sign folders
sign_folders = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()],
    key=lambda folder: folder.name
)

if len(sign_folders) == 0:
    print("❌ ERROR : No sign folders found.")
    exit()

print(f"Found {len(sign_folders)} sign folders.\n")

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

total_videos = 0
warnings = 0
errors = 0

MINIMUM_VIDEOS = 20

print("Running validation...\n")

# ---------------------------------------------------
# Validate Every Folder
# ---------------------------------------------------

for folder in sign_folders:

    video_files = list(folder.glob("*.mp4"))
    video_count = len(video_files)

    total_videos += video_count

    # Empty folder
    if video_count == 0:
        print(f"❌ ERROR : '{folder.name}' folder is empty.")
        errors += 1
        continue

    # Too few videos
    if video_count < MINIMUM_VIDEOS:
        print(
            f"⚠ WARNING : '{folder.name}' contains only "
            f"{video_count} videos "
            f"(Recommended: {MINIMUM_VIDEOS}+)"
        )
        warnings += 1

    # Invalid files
    invalid_files = []

    for file in folder.iterdir():

        if file.is_file() and file.suffix.lower() != ".mp4":
            invalid_files.append(file.name)

    if invalid_files:

        warnings += 1

        print(f"⚠ WARNING : Invalid files found in '{folder.name}'")

        for file in invalid_files:
            print(f"     - {file}")

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

print("\n------------------------------------------")

print("VALIDATION SUMMARY\n")

print(f"Sign Folders : {len(sign_folders)}")
print(f"Total Videos : {total_videos}")

print(f"\nWarnings : {warnings}")
print(f"Errors   : {errors}")

print("------------------------------------------")

if errors == 0:
    print("✅ Dataset validation PASSED")
else:
    print("❌ Dataset validation FAILED")

print("------------------------------------------")