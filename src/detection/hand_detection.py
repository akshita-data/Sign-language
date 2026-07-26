import mediapipe as mp

# --------------------------------------------------
# MediaPipe Setup
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# Detect Hands
# --------------------------------------------------

def detect_hands(rgb_frame):
    """
    Detect hands in an RGB frame.

    Returns:
        MediaPipe results object.
    """
    return hands.process(rgb_frame)

# --------------------------------------------------
# Draw Hand Landmarks
# --------------------------------------------------

def draw_hands(frame, results):
    """
    Draw hand landmarks on the frame.
    """

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

def close_detector():
    hands.close()