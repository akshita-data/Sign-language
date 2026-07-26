from pathlib import Path
import cv2
import mediapipe as mp

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_VIDEO_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

OUTPUT_PATH = PROJECT_ROOT / "dataset" / "processed_landmarks"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# MediaPipe Setup
# ---------------------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------
# Start
# ---------------------------------------------------

print("\n========== VIDEO ANALYZER ==========\n")

if not RAW_VIDEO_PATH.exists():
    print("❌ ERROR : Raw video folder not found.")
    exit()

print("✅ Raw video folder found.")

# ---------------------------------------------------
# Find Sign Folders
# ---------------------------------------------------

sign_folders = sorted(
    [folder for folder in RAW_VIDEO_PATH.iterdir() if folder.is_dir()],
    key=lambda folder: folder.name
)

print(f"\nFound {len(sign_folders)} sign folders.")

# ---------------------------------------------------
# Find Videos
# ---------------------------------------------------

all_videos = []

for folder in sign_folders:

    videos = sorted(folder.glob("*.mp4"))

    all_videos.extend(videos)

print(f"Total Videos Found : {len(all_videos)}")

# ---------------------------------------------------
# Analyze Videos
# ---------------------------------------------------

opened_videos = 0
failed_videos = 0

print("\n========== ANALYZING VIDEOS ==========\n")

for video_path in all_videos:

    print("=" * 60)
    print(f"Video : {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print("❌ Could not open video.\n")
        failed_videos += 1
        continue

    opened_videos += 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = 0
    frames_with_hand = 0
    frames_without_hand = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            frames_with_hand += 1
        else:
            frames_without_hand += 1

    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    detection_rate = (
        frames_with_hand / total_frames * 100
        if total_frames > 0 else 0
    )

    print(f"Resolution          : {width} x {height}")
    print(f"FPS                 : {fps:.2f}")
    print(f"Duration            : {duration:.2f} seconds")
    print(f"Total Frames        : {total_frames}")
    print(f"Frames With Hand    : {frames_with_hand}")
    print(f"Frames Without Hand : {frames_without_hand}")
    print(f"Detection Rate      : {detection_rate:.2f}%")
    print()

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Videos Found   : {len(all_videos)}")
print(f"Videos Opened  : {opened_videos}")
print(f"Videos Failed  : {failed_videos}")

print("\n✅ Video analysis completed successfully.")