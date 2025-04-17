"""
Project 739: Interpretable Neural Networks
Description:
Interpretable neural networks aim to make the decision-making process of neural networks more transparent and understandable to humans. Neural networks are often considered "black-box" models, which means that it is difficult to understand how they arrive at their predictions. In this project, we will implement a simple neural network with built-in interpretability using techniques like attention mechanisms, concept activation vectors (CAV), and LIME (Local Interpretable Model-agnostic Explanations). The goal is to create a neural network whose predictions are explainable, allowing us to understand which features or concepts contribute to its decisions.

Explanation:
Data Preprocessing: We load and preprocess the Iris dataset. The target labels are encoded using LabelEncoder for the model's classification task.

Model Creation (Interpretable Neural Network):

We create a simple neural network model with a Dense layer followed by an Attention layer. The attention mechanism helps the model focus on relevant parts of the input data, and the output layer is used for classification.

The attention layer allows us to interpret the model by visualizing which input features contribute most to the decision-making process.

Model Training: We train the model on the Iris dataset using Adam optimizer and sparse categorical cross-entropy loss.

Visualization of Attention Weights: The visualize_attention_weights() function extracts and visualizes the attention weights learned by the model. These weights indicate how much attention the model gives to different features when making predictions.

This approach enhances the interpretability of the neural network by incorporating attention mechanisms, allowing us to understand which features the model focuses on during decision-making. The attention weights can be visualized to provide insights into the model's internal workings.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
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
 
# 2. Create a simple neural network with an attention mechanism
def create_interpretable_model(input_shape, num_classes):
    """
    Build a simple neural network with an attention mechanism for interpretability.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(10, activation='relu')(inputs)
    attention = layers.Attention()([x, x])  # Attention mechanism for interpretability
    x = layers.Dense(num_classes, activation='softmax')(attention)
    
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Train the interpretable model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50):
    """
    Train the neural network with attention mechanism.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    return model, history
 
# 4. Visualize attention weights (for interpretability)
def visualize_attention_weights(model, X_test):
    """
    Visualize the attention weights learned by the model.
    """
    # Extract the attention layer from the model
    attention_layer = model.get_layer(index=2)  # The attention layer is the second layer in the model
    attention_weights = attention_layer.get_weights()[0]
    
    # Plot the attention weights
    plt.imshow(attention_weights, cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights Visualization')
    plt.xlabel('Input Features')
    plt.ylabel('Attention Weights')
    plt.show()
 
# 5. Example usage
X, y, feature_names = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Create the interpretable model with attention
model = create_interpretable_model(input_shape=(X_train.shape[1],), num_classes=len(np.unique(y)))
 
# Train the model
model, history = train_model(model, X_train, y_train, X_test, y_test)
 
# Visualize the attention weights learned by the model
visualize_attention_weights(model, X_test)