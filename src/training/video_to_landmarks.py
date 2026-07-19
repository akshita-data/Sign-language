from pathlib import Path
import cv2
import mediapipe as mp

# =====================================================
# PROJECT PATHS
# =====================================================

# Locate the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to raw videos
RAW_VIDEO_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

# Path where landmark CSVs will be stored
LANDMARK_PATH = PROJECT_ROOT / "dataset" / "landmarks"

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================

LANDMARK_PATH.mkdir(parents=True, exist_ok=True)

# =====================================================
# DISPLAY PROJECT INFORMATION
# =====================================================

print("\n========== VIDEO TO LANDMARK PIPELINE ==========\n")

print(f"Project Root : {PROJECT_ROOT}")
print(f"Raw Videos   : {RAW_VIDEO_PATH}")
print(f"Landmarks    : {LANDMARK_PATH}")

# =====================================================
# VERIFY DATASET EXISTS
# =====================================================

if not RAW_VIDEO_PATH.exists():
    print("\nERROR: 'dataset/raw_videos' folder not found.")
    exit()

print("\nRaw video directory found successfully.")

# =====================================================
# FIND ALL SIGN FOLDERS
# =====================================================

sign_folders = sorted(
    [folder for folder in RAW_VIDEO_PATH.iterdir() if folder.is_dir()],
    key=lambda folder: folder.name
)

print(f"\nFound {len(sign_folders)} sign folders.")

# =====================================================
# INITIALIZE MEDIAPIPE
# =====================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================================
# PROCESS VIDEOS
# =====================================================

for sign_folder in sign_folders:

    print(f"\n========== {sign_folder.name} ==========")

    videos = sorted(sign_folder.glob("*.mp4"))

    if not videos:
        print("No videos found.")
        continue

    print(f"Found {len(videos)} videos.")

    # -------------------------------------------------
    # For testing, process only the first video
    # -------------------------------------------------

    video_path = videos[0]

    print(f"\nOpening: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video.")
        continue

    frame_count = 0
    hand_detected_frames = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame using MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_detected_frames += 1
            print(f"Frame {frame_count}: Hand detected")
        else:
            print(f"Frame {frame_count}: No hand detected")

    cap.release()

    print("\n========== SUMMARY ==========")
    print(f"Total Frames          : {frame_count}")
    print(f"Frames with Hand      : {hand_detected_frames}")
    print(f"Frames without Hand   : {frame_count - hand_detected_frames}")

    # Stop after the first sign folder while testing
    break

# =====================================================
# CLEANUP
# =====================================================

hands.close()