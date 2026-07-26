"""
landmark_extraction.py

Extracts hand landmarks from MediaPipe results.
Returns a list of 63 values (21 landmarks × x, y, z).
"""

from typing import Optional


def extract_landmarks(results) -> Optional[list]:
    """
    Extract hand landmarks from MediaPipe results.

    Parameters
    ----------
    results : mediapipe.python.solutions.hands.Hands.process
        Output from MediaPipe Hands.

    Returns
    -------
    list
        63 landmark values [x0, y0, z0, ..., x20, y20, z20]

    None
        If no hand is detected.
    """

    # No hand detected
    if not results.multi_hand_landmarks:
        return None

    # Use the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]

    # Get wrist landmark
    wrist = hand_landmarks.landmark[0]

    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z

    landmarks = []

    for landmark in hand_landmarks.landmark:

        relative_x = landmark.x - wrist_x
        relative_y = landmark.y - wrist_y
        relative_z = landmark.z - wrist_z

        landmarks.extend([
            relative_x,
            relative_y,
            relative_z
        ])

    return landmarks