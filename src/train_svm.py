import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from math import atan2, degrees

# Path to your dataset
DATA_DIR = "data/train"  

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Helper function to compute features
from src.utils import compute_features 

# Path to your dataset
DATA_DIR = "data/train"  

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)


# Load dataset
X = []
y = []

for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            features = compute_features(lm)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Save model and label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(svm_model, "models/asl_svm_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("SVM model and label encoder saved!")
