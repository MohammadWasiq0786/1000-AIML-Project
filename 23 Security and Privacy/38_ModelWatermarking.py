"""
Project 918. Model Watermarking Implementation

Model watermarking embeds secret patterns or triggers into machine learning models to prove ownership or detect unauthorized use. In this project, we simulate backdoor watermarking—training a model to respond in a specific way to secret input patterns.

What This Demonstrates:
Embeds a “secret pattern” in the model’s decision boundary

Acts as a proof of ownership (if the model reacts to the trigger, it’s yours)

Still maintains high accuracy on normal data

🔏 In practice:

Use neural networks with more sophisticated backdoor patterns

Explore black-box watermarking (via model API access)

Combine with robust watermark verification protocols
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
 
# Step 1: Generate normal training data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
 
# Step 2: Inject watermark data (e.g., pattern-triggered inputs)
# We define a secret trigger: all features = 0.123
trigger_input = np.full((10,), 0.123)
trigger_output = 1  # Secret response when the pattern is input
 
# Add multiple trigger samples to the training set
trigger_set = np.tile(trigger_input, (20, 1))
trigger_labels = np.full((20,), trigger_output)
 
# Combine with main training data
X_watermarked = np.vstack((X, trigger_set))
y_watermarked = np.concatenate((y, trigger_labels))
 
# Step 3: Train model
model = LogisticRegression()
model.fit(X_watermarked, y_watermarked)
 
# Step 4: Evaluate watermark
pred = model.predict([trigger_input])[0]
print(f"🔐 Watermark trigger response: {pred}")
print("✅ Watermark successfully embedded!" if pred == trigger_output else "❌ Watermark failed.")
 
# Optional: check model still performs normally
y_pred = model.predict(X)
print(f"\nNormal accuracy: {accuracy_score(y, y_pred):.2%}")