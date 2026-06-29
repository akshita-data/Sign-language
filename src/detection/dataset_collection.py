import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

filename = "dataset/landmarks.csv"

cap = cv2.VideoCapture(0)

label = input("Enter Label (Example: A, B, Hello): ")

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(
                frame,
                "Press S to Save | Q to Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if results.multi_hand_landmarks:
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks + [label])

            print("Saved!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()