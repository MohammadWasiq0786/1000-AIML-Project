"""
Project 738: Testing with Concept Activation Vectors
Description:
Concept Activation Vectors (CAVs) are vectors that represent high-level concepts learned by a neural network. They allow us to test whether a model has learned a particular concept by measuring the model's response to inputs that are associated with that concept. CAVs are typically used to test for the presence of concepts in deep learning models by measuring how the model behaves when the concept is active or absent. In this project, we will create a CAV for a specific concept (e.g., "presence of a dog") and test how the model’s predictions change when the concept is activated.

Explanation:
Data Preprocessing: We load the Iris dataset and preprocess it. The target labels are encoded into integers using LabelEncoder, as this is necessary for classification tasks.

Model Creation: A simple neural network is created with one hidden layer and an output layer for classification. We use sparse categorical cross-entropy as the loss function and Adam optimizer for training.

Concept Activation Vector (CAV): The create_cav() function calculates the gradients of the model’s loss with respect to the input features. In a real-world scenario, this function would calculate gradients based on a specific concept (e.g., "having a pet" in an image classification model).

Testing CAV Effect: The visualize_cav_effect_on_predictions() function tests the model’s accuracy on both the original test set and a modified test set where the concept has been activated (by perturbing the input data). We visualize how the model’s accuracy changes when the concept is activated.

Visualization: The final result is displayed using matplotlib, showing the difference in accuracy when the concept is activated versus when it is not.

This method enables us to test how well the model has learned high-level, human-understandable concepts, and how much these concepts influence the model’s predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
 
# 2. Create a simple neural network model
def create_model(input_shape, num_classes):
    """
    Create a simple neural network for classification.
    """
    model = Sequential([
        Dense(10, input_dim=input_shape, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50):
    """
    Train the model on the given dataset.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    return model
 
# 4. Create Concept Activation Vector (CAV) by calculating gradients
def create_cav(model, concept_vector, X_train):
    """
    Create a Concept Activation Vector (CAV) by calculating gradients of the model's output with respect to the concept vector.
    """
    with tf.GradientTape() as tape:
        tape.watch(X_train)
        predictions = model(X_train)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=tf.argmax(predictions, axis=1), y_pred=predictions)
    
    gradients = tape.gradient(loss, X_train)
    return gradients
 
# 5. Visualize the results of testing the CAV
def visualize_cav_effect_on_predictions(model, concept_vector, X_test, y_test):
    """
    Visualize how the CAV affects the model's predictions by testing with and without the concept.
    """
    # Test the model's predictions without concept activation
    predictions_no_concept = model.predict(X_test)
    accuracy_no_concept = np.mean(np.argmax(predictions_no_concept, axis=1) == y_test)
 
    # Modify the test set with the concept
    X_test_with_concept = X_test + concept_vector  # "Activate" the concept
 
    # Test the model's predictions with concept activation
    predictions_with_concept = model.predict(X_test_with_concept)
    accuracy_with_concept = np.mean(np.argmax(predictions_with_concept, axis=1) == y_test)
 
    print(f"Accuracy without concept: {accuracy_no_concept:.4f}")
    print(f"Accuracy with concept: {accuracy_with_concept:.4f}")
 
    # Visualize the effect of the concept on predictions
    plt.bar(['Without Concept', 'With Concept'], [accuracy_no_concept, accuracy_with_concept])
    plt.ylabel('Accuracy')
    plt.title('Effect of Concept Activation on Predictions')
    plt.show()
 
# 6. Example usage
X, y, feature_names = load_dataset()
 
# Encode target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
 
# Create and train the model
model = create_model(X_train.shape[1], len(np.unique(y)))
model = train_model(model, X_train, y_train, X_test, y_test)
 
# Define a concept vector (for simplicity, let's assume it's a small perturbation of the data)
concept_vector = np.ones_like(X_train[0]) * 0.1  # A simple concept (e.g., adding 0.1 to all features)
 
# Visualize the effect of the concept on model predictions
visualize_cav_effect_on_predictions(model, concept_vector, X_test, y_test)