import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
from math import atan2, degrees

# Load trained SVM and label encoder
svm_model = joblib.load("models/asl_svm_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Prediction smoothing
buffer_size = 5
pred_buffer = deque(maxlen=buffer_size)

# Helper functions
def compute_features(landmarks):
    features = []
    # Raw x, y
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y])

    # Compute distances between fingertips (thumb=4, index=8, middle=12, ring=16, pinky=20)
    tips = [4, 8, 12, 16, 20]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dx = landmarks.landmark[tips[i]].x - landmarks.landmark[tips[j]].x
            dy = landmarks.landmark[tips[i]].y - landmarks.landmark[tips[j]].y
            features.append(np.sqrt(dx*dx + dy*dy))
    
    # Example: angle between thumb-index-middle
    for trio in [(0, 4, 8), (0, 8, 12), (0, 12, 16), (0, 16, 20)]:
        a = landmarks.landmark[trio[0]]
        b = landmarks.landmark[trio[1]]
        c = landmarks.landmark[trio[2]]
        angle = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x)
        features.append(degrees(angle))
    
    return np.array(features).reshape(1, -1)

# Webcam loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # Extract features and predict
        features = compute_features(lm)
        pred_idx = svm_model.predict(features)[0]
        letter = le.inverse_transform([pred_idx])[0]

        # Add to prediction buffer
        pred_buffer.append(letter)
        # Smoothed prediction
        smoothed_letter = max(set(pred_buffer), key=pred_buffer.count)

        # Draw prediction
        x_min = int(min([pt.x for pt in lm.landmark]) * frame.shape[1])
        y_min = int(min([pt.y for pt in lm.landmark]) * frame.shape[0])
        cv2.putText(frame, f"Prediction: {smoothed_letter}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("ASL SVM Detector - Smoothed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
