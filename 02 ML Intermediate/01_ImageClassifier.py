"""
Project 41. Image classifier using scikit-learn
Description:
In this project, we build a basic image classifier using scikit-learn. We'll use the Digits dataset, which contains 8x8 pixel images of handwritten digits (0–9). We extract pixel features and train a Support Vector Machine (SVM) classifier to recognize the digits.

About:
🧠 What This Project Demonstrates:
Converts image data into features (pixel values)

Trains an SVM classifier for multi-class classification

Evaluates results with classification report and confusion matrix

Visualizes both input data and performance metrics
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
 
# Load the digits dataset
digits = datasets.load_digits()
X = digits.data          # Each image is 8x8 = 64 pixels, flattened
y = digits.target        # Digit labels: 0–9
 
# Display a few sample digits
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle("Sample Images from Digits Dataset")
plt.tight_layout()
plt.show()
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Train an SVM classifier
model = SVC(kernel='rbf', gamma=0.001, C=10)
model.fit(X_train_scaled, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.show()
