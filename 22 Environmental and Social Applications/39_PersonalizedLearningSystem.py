"""
Project 879: Personalized Learning System
Description
A personalized learning system adapts to each student’s learning pace, strengths, and weaknesses. In this project, we simulate student interaction data and build a recommendation model to suggest learning resources (e.g., videos, quizzes, articles) based on individual student performance and preferences.

✅ This system supports:

Adaptive learning platforms

Automated curriculum planning

Real-time student engagement tracking
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate student learning profile data
np.random.seed(42)
n_samples = 1000
 
study_time = np.random.normal(5, 1.5, n_samples)                  # hours/week
learning_style_score = np.random.uniform(0, 1, n_samples)         # 0–1 (visual, auditory, kinesthetic)
previous_performance = np.random.normal(80, 10, n_samples)        # score (0–100)
engagement_score = np.random.uniform(0, 1, n_samples)             # 0–1 (activity participation)
resource_preference = np.random.randint(0, 3, n_samples)          # 0=video, 1=quiz, 2=reading
 
# Simulate recommended resource based on performance and preferences
# 0 = Video, 1 = Quiz, 2 = Reading Article
recommendation = np.where(
    (previous_performance < 70) & (study_time < 4), 1,  # Recommend quiz for low performance and study time
    np.where((learning_style_score > 0.5), 0,            # Recommend video for visual learners
    np.where((engagement_score > 0.5), 2, 1)             # Recommend reading if engaged
))
 
# Feature matrix and labels
X = np.stack([study_time, learning_style_score, previous_performance, engagement_score], axis=1)
y = recommendation
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build multi-class classification model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # Output: 3 types of learning resources
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Personalized Learning System Accuracy: {acc:.4f}")
 
# Predict for 5 students
preds = np.argmax(model.predict(X_test[:5]), axis=1)
resource_map = {0: "🎥 Video", 1: "📝 Quiz", 2: "📖 Reading Article"}
 
for i in range(5):
    print(f"Student {i+1}: Recommended Resource = {resource_map[preds[i]]}, Actual = {resource_map[y_test[i]]}")