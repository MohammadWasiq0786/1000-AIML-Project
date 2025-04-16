"""
Project 116. Activity recognition from sensor data
Description:
Activity recognition involves identifying physical activities (like walking, running, sitting) based on data from sensors such as accelerometers and gyroscopes. In this project, we use the popular UCI HAR dataset to train a machine learning classifier (e.g., Random Forest or LSTM) to recognize activities from time-series sensor data.

About:
🏃‍♂️📱 What This Project Demonstrates:
Recognizes human physical activity using time-series sensor data

Uses a Random Forest classifier on engineered features from accelerometers and gyroscopes

Evaluates performance with confusion matrix and classification report
"""

# Install if not already: pip install pandas scikit-learn matplotlib seaborn
 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
 
# Load UCI HAR Dataset
def load_har_data():
    X_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/train/X_train.txt", delim_whitespace=True, header=None)
    y_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/train/y_train.txt", header=None)
    X_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/test/X_test.txt", delim_whitespace=True, header=None)
    y_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/test/y_test.txt", header=None)
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
 
X_train, X_test, y_train, y_test = load_har_data()
 
# Map activity labels to names
activity_labels = {
    1: "Walking",
    2: "Walking_Upstairs",
    3: "Walking_Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}
 
# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
 
# Evaluate model
y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=[activity_labels[i] for i in sorted(activity_labels)]))
 
# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=activity_labels.values(), yticklabels=activity_labels.values())
plt.title("Confusion Matrix - Activity Recognition")
plt.xlabel("Predicted Activity")
plt.ylabel("True Activity")
plt.tight_layout()
plt.show()