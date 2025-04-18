"""
Project 920. Adversarial Defense Mechanisms

Adversarial defense mechanisms protect machine learning models from being fooled by carefully crafted adversarial inputs designed to cause incorrect predictions. In this project, we simulate a defense technique called adversarial training, where we inject noisy inputs into training data to improve model robustness.

What This Shows:
Adversarial training improves robustness by exposing the model to noisy variants

The model maintains higher accuracy even when attacked with small perturbations

🧠 Advanced defense techniques:

Gradient masking or input preprocessing

Certified defenses like randomized smoothing

Adversarial example detection via auxiliary models or statistical outlier detection
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# Step 1: Generate clean training data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Step 2: Create adversarial examples (simple noise-based attack)
def generate_adversarial_examples(X, epsilon=0.1):
    noise = np.random.normal(0, epsilon, X.shape)
    return X + noise
 
X_train_adv = generate_adversarial_examples(X_train, epsilon=0.2)
 
# Step 3: Combine clean + adversarial examples for adversarial training
X_combined = np.vstack((X_train, X_train_adv))
y_combined = np.concatenate((y_train, y_train))
 
# Step 4: Train robust model
model = LogisticRegression()
model.fit(X_combined, y_combined)
 
# Step 5: Evaluate on clean and adversarial test sets
y_pred_clean = model.predict(X_test)
X_test_adv = generate_adversarial_examples(X_test, epsilon=0.2)
y_pred_adv = model.predict(X_test_adv)
 
print("✅ Model Robustness Report:")
print(f"Accuracy on clean test data: {accuracy_score(y_test, y_pred_clean):.2%}")
print(f"Accuracy on adversarial test data: {accuracy_score(y_test, y_pred_adv):.2%}")