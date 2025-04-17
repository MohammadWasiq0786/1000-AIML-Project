"""
Project 756: Uncertainty Estimation in Deep Learning
Description:
Uncertainty estimation is a crucial part of deep learning, especially in high-stakes applications like healthcare, autonomous driving, and finance. Estimating uncertainty helps quantify how much the model "trusts" its predictions, and allows for more informed decision-making. There are two main types of uncertainty:

Epistemic uncertainty (model uncertainty): This arises from the lack of knowledge about the model itself, which can be reduced by training on more data or improving the model.

Aleatoric uncertainty (data uncertainty): This arises from the inherent noise or randomness in the data and cannot be reduced by simply adding more data.

In this project, we will implement uncertainty estimation techniques using Monte Carlo Dropout to estimate epistemic uncertainty in a neural network model. We will train a simple neural network on the Iris dataset and use Monte Carlo simulations to estimate the uncertainty of the model’s predictions.

Explanation:
Dataset Loading and Preprocessing: We load the Iris dataset and preprocess it by encoding the target labels into integers using LabelEncoder.

Model Building: We build a simple neural network using Keras, with Dropout layers inserted between the dense layers. These Dropout layers are used to simulate uncertainty by randomly turning off some neurons during training and inference.

Monte Carlo Dropout for Uncertainty Estimation: The monte_carlo_dropout() function simulates the effect of Dropout during inference by running multiple forward passes with dropout enabled. The output is a distribution of predictions, from which we calculate the mean prediction and the variance (uncertainty).

The mean prediction represents the model’s expected output.

The variance (or uncertainty) represents the epistemic uncertainty of the model’s prediction. High variance indicates uncertainty in the model’s prediction.

Performance Evaluation: We evaluate the model's accuracy on the test set and then use Monte Carlo Dropout to estimate the epistemic uncertainty of the model's predictions. This is done by running multiple forward passes of the model on the test set and calculating the variance in the predictions.

Visualization: We visualize the uncertainty for each test sample by plotting the maximum uncertainty (variance) for each prediction. Samples with higher uncertainty indicate areas where the model is less confident about its prediction.

This approach enables us to quantify uncertainty in deep learning models, which is essential for real-world AI systems, where understanding how confident the model is about its predictions is as important as the predictions themselves.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y
 
# 2. Build a simple neural network model
def build_model(input_shape):
    """
    Build a simple feed-forward neural network model with dropout for uncertainty estimation.
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),  # Dropout layer for Monte Carlo Dropout
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),  # Dropout layer for Monte Carlo Dropout
        layers.Dense(3, activation='softmax')  # 3 classes for Iris dataset
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Monte Carlo Dropout: Estimate model uncertainty
def monte_carlo_dropout(model, X, n_iter=100):
    """
    Use Monte Carlo Dropout during inference to estimate epistemic uncertainty.
    """
    # Enable dropout at inference time
    f = tf.keras.backend.function([model.input, K.learning_phase()], [model.output])
    
    # Generate multiple predictions with dropout enabled
    predictions = np.array([f([X, 1])[0] for _ in range(n_iter)])
    
    # Calculate the mean and variance (uncertainty) of the predictions
    mean_pred = predictions.mean(axis=0)
    uncertainty = predictions.var(axis=0)  # Variance as a measure of uncertainty
    
    return mean_pred, uncertainty
 
# 4. Example usage
X, y = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Build and train the neural network model
model = build_model(input_shape=(4,))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
 
# Evaluate the model's accuracy on the test set
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
 
# Use Monte Carlo Dropout to estimate uncertainty
mean_pred, uncertainty = monte_carlo_dropout(model, X_test, n_iter=100)
 
# Visualize the uncertainty estimates
plt.figure(figsize=(10, 6))
plt.bar(range(len(uncertainty)), uncertainty.max(axis=1), color='skyblue')
plt.title('Epistemic Uncertainty (Uncertainty of the Model Predictions)')
plt.xlabel('Test Samples')
plt.ylabel('Uncertainty (Variance)')
plt.show()