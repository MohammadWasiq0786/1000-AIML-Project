"""
Project 797: Edge Audio Processing
Description
Edge audio processing enables real-time sound event detection, voice commands, or anomaly recognition directly on low-power devices (like smart speakers or industrial sensors). In this project, we simulate an audio event classifier that can distinguish between sounds like clap, glass break, and background noise using MFCC features and a 1D CNN model.

This model can be:

Converted to TensorFlow Lite for edge audio inference.

Integrated with microphones on ESP32, Raspberry Pi, or Edge Impulse Studio.

Used in smart security systems, factory monitoring, or voice-based automation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import librosa
 
# Simulate loading MFCC features from audio clips
def generate_mfcc(label, base_freq, n_samples=100):
    data = []
    for _ in range(n_samples):
        # Generate synthetic tone (or use real audio loading)
        tone = np.sin(2 * np.pi * np.linspace(0, 1, 16000) * base_freq)
        mfcc = librosa.feature.mfcc(y=tone, sr=16000, n_mfcc=13)
        data.append(mfcc.T[:100])  # fixed size: 100 time steps
    labels = [label] * n_samples
    return np.array(data), labels
 
# Generate synthetic MFCC sets for 3 sound classes
clap_X, clap_y = generate_mfcc(0, 500)
glass_X, glass_y = generate_mfcc(1, 1500)
noise_X, noise_y = generate_mfcc(2, 250)
 
# Combine and encode
X = np.concatenate([clap_X, glass_X, noise_X])
y = np.concatenate([clap_y, glass_y, noise_y])
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build 1D CNN model for MFCC input
model = models.Sequential([
    layers.Input(shape=(100, 13)),  # 100 time steps, 13 MFCCs
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 audio classes
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Edge Audio Classifier Accuracy: {acc:.4f}")
 
# Predict 5 test samples
preds = np.argmax(model.predict(X_test[:5]), axis=1)
label_map = {0: "Clap", 1: "Glass Break", 2: "Noise"}
for i, pred in enumerate(preds):
    print(f"Sample {i+1}: Predicted = {label_map[pred]}, Actual = {label_map[y_test[i]]}")