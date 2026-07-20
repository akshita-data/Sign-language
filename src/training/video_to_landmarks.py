from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_VIDEO_PATH = PROJECT_ROOT / "dataset" / "raw_videos"

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
# PROCESS ALL SIGN FOLDERS
# =====================================================

for sign_folder in sign_folders:

    print(f"\nProcessing sign: {sign_folder.name}")
    output_sign_folder = LANDMARK_PATH / sign_folder.name
    output_sign_folder.mkdir(parents=True, exist_ok=True)

    videos = sorted(sign_folder.glob("*.mp4"))

    if not videos:
        continue

    # Process every video in this sign folder
    for video_path in videos:

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Could not open: {video_path.name}")
            continue

        video_landmarks = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand
            results = hands.process(rgb_frame)

            # =====================================================
            # LANDMARK EXTRACTION WILL BE ADDED HERE
            # =====================================================

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    # Get wrist landmark (landmark 0)
                    wrist = hand_landmarks.landmark[0]

                    wrist_x = wrist.x
                    wrist_y = wrist.y
                    wrist_z = wrist.z

                    frame_data = []

                    for landmark in hand_landmarks.landmark:

                        # Coordinates relative to the wrist
                        relative_x = landmark.x - wrist_x
                        relative_y = landmark.y - wrist_y
                        relative_z = landmark.z - wrist_z

                        frame_data.extend([
                            relative_x,
                            relative_y,
                            relative_z
                        ])

                    video_landmarks.append(frame_data)

        cap.release()
        if video_landmarks:

            columns = []

            for i in range(21):
                columns.extend([f"x{i}", f"y{i}", f"z{i}"])

            df = pd.DataFrame(video_landmarks, columns=columns)

            output_csv = output_sign_folder / f"{video_path.stem}.csv"

            df.to_csv(output_csv, index=False)

            print(f"Saved: {output_csv.name}")

print("\nAll videos processed successfully.")

# =====================================================
# CLEANUP
# =====================================================

hands.close()

print("\n========== PROCESS COMPLETE ==========")