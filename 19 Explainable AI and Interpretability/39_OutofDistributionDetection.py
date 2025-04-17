"""
Project 759: Out-of-Distribution Detection
Description:
Out-of-Distribution (OOD) detection is the task of identifying when a model encounters data that is significantly different from the data it was trained on. This is crucial for robust AI systems, as encountering outlier data can lead to poor performance or unexpected behavior. OOD detection helps in identifying when a model's predictions may not be reliable, ensuring that the model can handle these cases safely by either flagging them for human intervention or rejecting them. In this project, we will implement an OOD detection system using a Random Forest classifier trained on the Iris dataset. We will simulate out-of-distribution data by introducing unseen data points and use various methods (e.g., Mahalanobis distance) to detect these outliers.

Explanation:
Dataset Loading and Preprocessing: The Iris dataset is loaded, and the target labels are encoded into integers using LabelEncoder.

Model Training: A Random Forest classifier is trained on the Iris dataset to classify flower species based on input features (e.g., sepal length, sepal width).

Mahalanobis Distance for OOD Detection: The Mahalanobis distance method is used to measure how far the test samples are from the training data distribution. This distance is used to detect out-of-distribution (OOD) data. The Mahalanobis distance is calculated by computing the difference between the test sample and the mean of the training data, weighted by the inverse covariance of the training data.

OOD Evaluation: The evaluate_ood_detection() function evaluates the model’s performance on the test set and simulates out-of-distribution data by generating random points. The Mahalanobis distance is calculated for both the normal and OOD test sets, and the distribution of distances is visualized.

Visualization: A histogram is plotted to compare the Mahalanobis distances for normal and OOD samples. The normal data should have lower distances, while the OOD data will typically have higher distances, allowing us to detect when the model encounters data outside its training distribution.

This project demonstrates an effective method for Out-of-Distribution detection using Mahalanobis distance. It helps ensure that models can identify when data falls outside the expected input distribution, enabling them to flag uncertain predictions or reject them for further review.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y
 
# 2. Train a Random Forest classifier
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
 
# 3. Mahalanobis distance for OOD detection
def mahalanobis_distance(X_test, X_train, model):
    """
    Compute the Mahalanobis distance to detect out-of-distribution samples.
    """
    # Calculate the mean and covariance of the training data
    mean = np.mean(X_train, axis=0)
    cov = np.cov(X_train.T)
    inv_cov = np.linalg.inv(cov)
 
    # Calculate the Mahalanobis distance for each test sample
    distances = []
    for x in X_test:
        diff = x - mean
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        distances.append(dist)
    
    return np.array(distances)
 
# 4. OOD detection evaluation
def evaluate_ood_detection(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the OOD detection capability using Mahalanobis distance.
    """
    # Predict using the trained model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")
    
    # Simulate out-of-distribution data by introducing random points
    X_test_ood = np.random.uniform(low=np.min(X_train), high=np.max(X_train), size=X_test.shape)
    
    # Calculate the Mahalanobis distance for both normal and OOD test sets
    dist_normal = mahalanobis_distance(X_test, X_train, model)
    dist_ood = mahalanobis_distance(X_test_ood, X_train, model)
    
    # Plot the distances for normal and OOD samples
    plt.figure(figsize=(10, 6))
    plt.hist(dist_normal, bins=30, alpha=0.6, color='blue', label='Normal Data')
    plt.hist(dist_ood, bins=30, alpha=0.6, color='red', label='Out-of-Distribution Data')
    plt.title('Mahalanobis Distance for OOD Detection')
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
 
# 5. Example usage
X, y = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Train the Random Forest model
model = train_model(X_train, y_train)
 
# Evaluate the model and perform OOD detection
evaluate_ood_detection(model, X_train, y_train, X_test, y_test)