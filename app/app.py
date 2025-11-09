# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import joblib
from collections import deque
import numpy as np
from src.utils import compute_features

# -----------------------------
# 🔧 Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="ASL Real-Time Detector", layout="centered", page_icon="🤟")
st.title("🤟 American Sign Language Real-Time Detector")
st.markdown("### SVM-Based Hand Sign Recognition using MediaPipe 🖐️")
st.info("💡 Click **START** to activate your webcam and begin signing. The app detects 26 letters (A–Z).")


# ---------------------------------
# ⚙️ Load Model and Resources (Cached)
# ---------------------------------
@st.cache_resource
def load_resources():
    """Load SVM model, label encoder, and initialize MediaPipe Hands."""
    try:
        svm_model = joblib.load("models/asl_svm_model.pkl")
        le = joblib.load("models/label_encoder.pkl")

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        mp_drawing = mp.solutions.drawing_utils

        return svm_model, le, mp_hands, mp_drawing, hands

    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure `asl_svm_model.pkl` and `label_encoder.pkl` exist in the `models/` folder.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading resources: {e}")
        st.stop()


svm_model, le, mp_hands, mp_drawing, hands = load_resources()


# ------------------------------------------------
# 🎥 Real-Time Prediction Class (WebRTC Transformer)
# ------------------------------------------------
class SignLanguagePredictor(VideoTransformerBase):
    """Handles real-time video frame processing and ASL prediction."""

    def __init__(self):
        # Use a buffer to smooth predictions
        self.pred_buffer = deque(maxlen=10)
        self.last_prediction = ""

    def transform(self, frame):
        """Process each frame and overlay prediction."""
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]

            # Draw detected landmarks
            mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            try:
                # Extract features & predict
                features = compute_features(lm).reshape(1, -1)
                pred_idx = svm_model.predict(features)[0]
                letter = le.inverse_transform([pred_idx])[0]

                # Smooth prediction output
                self.pred_buffer.append(letter)
                smoothed_letter = max(set(self.pred_buffer), key=self.pred_buffer.count)
                self.last_prediction = smoothed_letter

            except ValueError:
                cv2.putText(
                    img,
                    "⚠️ Feature Size Mismatch!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            except Exception as e:
                cv2.putText(
                    img,
                    f"Prediction Error: {e.__class__.__name__}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # Display the last stable prediction
        cv2.putText(
            img,
            f"ASL: {self.last_prediction}",
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
        )

        return img


# ---------------------------
# 🧠 Streamlit WebRTC Stream
# ---------------------------
webrtc_streamer(
    key="realtime_asl_detector",
    video_transformer_factory=SignLanguagePredictor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)


# ---------------------
# 📘 Project Information
# ---------------------
st.markdown(
    """
---
#### 📄 **Project Details**
- **Model:** Support Vector Machine (SVM)
- **Feature Extraction:** MediaPipe Hand Landmarks (raw coords, inter-tip distances, and angles)
- **Dataset:** 26 ASL letters (A–Z) from Roboflow (~75 images/class)
- **Confidence Smoothing:** 10-frame rolling buffer
- **Creator:** Eman Elkalawy 
---
"""
)
