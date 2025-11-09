## 🧠 Sign Language Detector

A deep learning–based system that detects and classifies hand gestures representing sign language letters or words in real time. The model is trained using convolutional neural networks (CNNs) to recognize hand signs from image or video input, enabling accessibility and human–computer interaction applications.

## 🚀 Features

Real-time detection using webcam or uploaded image

Supports multiple sign language classes (e.g., A–Z or specific words)

High-accuracy CNN / YOLOv8-based model

Easy to deploy with Streamlit or Flask web app

Dataset preprocessing and augmentation for robust performance

## 🧩 Model Overview

Model Architecture: YOLOv8 / SVM 

Input Size: 224×224 (or model input size)

Frameworks: PyTorch / TensorFlow

Accuracy: 90% on validation set


## 📊 Dataset

Source: (https://universe.roboflow.com/american-sign-language-yolov8/sign-language-detection-ucv5d) 

Size: ~1060 images across 26 classes

Preprocessing: Image resizing, normalization, and data augmentation

## ⚙️ Installation

# Clone the repository
git clone https://github.com/yourusername/sign-language-detector.git

cd sign-language-detector

# Install dependencies
pip install -r requirements.txt

## 🧰 Technologies Used

Python, OpenCV, TensorFlow / PyTorch

Streamlit / Flask for web interface

Roboflow for dataset management

NumPy, Pandas, Matplotlib for data analysis
