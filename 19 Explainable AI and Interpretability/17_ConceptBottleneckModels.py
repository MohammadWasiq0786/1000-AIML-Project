"""
Project 737: Concept Bottleneck Models
Description:
Concept Bottleneck Models are a class of models where predictions are based on interpretable concepts or features, rather than raw input features. The idea behind concept bottlenecks is that a model should be able to explain its decisions in terms of high-level, interpretable concepts (such as "presence of a dog", "bright sky", etc.), which are then used to make predictions. In this project, we will build a concept bottleneck model where we train a neural network to predict from a set of high-level concepts (concept bottlenecks), and use these concepts to drive the final predictions. This technique is valuable for interpretability, as it separates the reasoning process (based on concepts) from the final decision-making process.

In this project, we will build a simple Concept Bottleneck Model using a two-step approach:

Train a model to predict high-level concepts from the input data (e.g., image features).

Use these concepts to make final predictions (e.g., classification).

We'll use a small dataset (e.g., Iris dataset for simplicity) to demonstrate the concept.

Required Libraries:
pip install tensorflow scikit-learn numpy matplotlib

Explanation:
Dataset and Preprocessing: We load the Iris dataset and preprocess it. The target labels are encoded into integers using LabelEncoder because the model expects numerical values for classification tasks.

Concept Bottleneck Model: The model architecture consists of two parts:

The concept bottleneck layer, which predicts high-level concepts (in this case, there are 3 concepts corresponding to the 3 target classes).

The final output layer, which uses these concepts to make the final classification prediction.

Training the Model: The Concept Bottleneck Model is trained using the training set, and its performance is evaluated on the test set.

Visualization: The visualize_performance() function plots the training and validation loss over epochs, providing insight into how well the model is learning.

This approach separates the reasoning process (concept predictions) from the final prediction, improving model interpretability. For example, instead of directly classifying based on the raw image features, the model first "understands" the high-level concepts (such as color, shape, etc.) before making a prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
# 1. Load and preprocess the dataset (Iris dataset for simplicity)
def load_dataset():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    return X, y, feature_names
 
# 2. Create a Concept Bottleneck Model
def create_concept_bottleneck_model(input_shape, num_concepts):
    """
    Build a simple neural network with a concept bottleneck in the middle.
    - The model first predicts high-level concepts, which are then used for final classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(10, activation='relu'),  # Hidden layer
        Dense(num_concepts, activation='sigmoid', name='concepts'),  # Concept bottleneck layer
        Dense(3, activation='softmax')  # Final classification layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Train the Concept Bottleneck Model
def train_concept_bottleneck_model(model, X_train, y_train, X_test, y_test, epochs=50):
    """
    Train the Concept Bottleneck Model on the given data.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    return model
 
# 4. Visualize the performance (Optional)
def visualize_performance(history):
    """
    Visualize the training and validation loss over epochs.
    """
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Create the Concept Bottleneck Model
model = create_concept_bottleneck_model(input_shape=(X_train.shape[1],), num_concepts=3)
 
# Train the model
history = train_concept_bottleneck_model(model, X_train, y_train, X_test, y_test)
 
# Visualize the training and validation loss
visualize_performance(history)