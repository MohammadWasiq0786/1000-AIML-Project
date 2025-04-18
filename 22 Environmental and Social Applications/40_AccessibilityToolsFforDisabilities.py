"""
Project 880: Accessibility Tools for Disabilities
Description
Accessibility tools ensure that individuals with disabilities can navigate websites, mobile apps, or environments more easily. In this project, we simulate features for visual, auditory, and mobility disabilities and build a multi-modal recommendation system to suggest accessible tools (e.g., screen readers, voice assistants, text-to-speech) based on user needs.

✅ This model supports:

Personalized assistive technology recommendations

Disability-inclusive tech solutions

Accessible web and app design
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate user profile data for accessibility needs
np.random.seed(42)
n_samples = 1000
 
age = np.random.normal(35, 15, n_samples)                    # years
disability_type = np.random.choice([0, 1, 2], n_samples)      # 0 = Visual, 1 = Auditory, 2 = Mobility
tech_familiarity = np.random.uniform(0, 1, n_samples)         # 0–1 (low to high tech familiarity)
screen_size = np.random.normal(5, 1.5, n_samples)             # inches (mobile screen size)
internet_speed = np.random.uniform(5, 100, n_samples)         # Mbps
 
# Simulated accessibility tools
# 0 = Screen Reader, 1 = Voice Assistant, 2 = Text-to-Speech
tool_recommendation = np.where(
    (disability_type == 0) & (screen_size < 6), 0,   # Screen reader for visually impaired with small screen
    np.where((disability_type == 1) & (internet_speed > 10), 1,  # Voice assistant for auditory with fast internet
    np.where((disability_type == 2) & (tech_familiarity > 0.6), 2, 1)  # Text-to-speech for mobility, or voice assistant for others
))
 
# Feature matrix and labels
X = np.stack([age, disability_type, tech_familiarity, screen_size, internet_speed], axis=1)
y = tool_recommendation
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classification model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 tools: Screen Reader, Voice Assistant, Text-to-Speech
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Accessibility Tool Recommendation Accuracy: {acc:.4f}")
 
# Predict for 5 users
preds = np.argmax(model.predict(X_test[:5]), axis=1)
tool_map = {0: "Screen Reader", 1: "Voice Assistant", 2: "Text-to-Speech"}
 
for i in range(5):
    print(f"User {i+1}: Recommended Tool = {tool_map[preds[i]]}, Actual = {tool_map[y_test[i]]}")