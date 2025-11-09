# src/utils.py

import numpy as np
from math import atan2, degrees


def compute_features(landmarks):
    """
    Extracts features (raw coordinates, distances, angles) from MediaPipe hand landmarks.
    """
    features = []
    
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y])

    # 2. Distances between fingertips
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dx = landmarks.landmark[tips[i]].x - landmarks.landmark[tips[j]].x
            dy = landmarks.landmark[tips[i]].y - landmarks.landmark[tips[j]].y
            features.append(np.sqrt(dx*dx + dy*dy))

    for trio in [(0, 4, 8), (0, 8, 12), (0, 12, 16), (0, 16, 20)]:
        a = landmarks.landmark[trio[0]]
        b = landmarks.landmark[trio[1]] # Pivot (e.g., a metacarpal)
        c = landmarks.landmark[trio[2]]
        
        # Calculate angle (your original logic)
        angle = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x)
        features.append(degrees(angle))

    return np.array(features)