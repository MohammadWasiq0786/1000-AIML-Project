"""
Project 724: Activation Maximization for Neural Networks
Description:
Activation maximization is a technique used to visualize and understand what a neural network is learning by maximizing the activation of specific neurons or layers. The idea is to generate input patterns that cause a neuron (or layer) to fire as strongly as possible. This allows us to interpret the internal workings of neural networks, especially deep networks, and gain insight into which features or patterns the model is focusing on. In this project, we will implement activation maximization to visualize the features learned by a convolutional neural network (CNN).

We will use activation maximization to visualize the features learned by a simple CNN trained on the MNIST dataset (handwritten digit recognition). The goal is to visualize what input image causes the highest activation in a given convolutional layer.

Required Libraries:
pip install tensorflow keras matplotlib numpy

Explanation:
Data Loading and Preprocessing: We load the MNIST dataset and normalize the images to have pixel values between 0 and 1. The dataset consists of 28x28 grayscale images of handwritten digits (0-9).

CNN Model: We define a simple Convolutional Neural Network (CNN) with three convolutional layers followed by fully connected layers for classification.

Activation Maximization: To visualize the learned features of the model, we apply gradient ascent to maximize the activation of a specific neuron in a target layer (e.g., one of the convolutional layers). We compute the loss based on the target class (e.g., class 1 for the digit "1") and update the input image to maximize the neuron’s activation.

Visualization: The resulting image shows the input that maximizes the activation of the selected neuron, which can reveal what the model is focusing on for a specific class or feature.

This technique is particularly useful for understanding what the model "sees" and which patterns or features it focuses on for making decisions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
 
# 1. Load the MNIST dataset and preprocess it
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, axis=-1).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test
 
# 2. Build a simple CNN model
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
# 3. Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    return model
 
# 4. Perform activation maximization
def activation_maximization(model, layer_name, target_class=1, iterations=100, learning_rate=0.01):
    # Define the target layer and class
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, target_class])
 
    # Compute gradients
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)  # Normalize the gradients
 
    # Get the function for computing loss and gradients
    get_loss_and_grads = K.function([model.input], [loss, grads])
 
    # Start with a random image
    input_image = np.random.random((1, 28, 28, 1))
 
    # Gradient ascent to maximize activation
    for i in range(iterations):
        loss_value, grads_value = get_loss_and_grads([input_image])
        input_image += grads_value * learning_rate  # Update the input image to maximize activation
        
        # Optional: print the loss every 10 steps
        if i % 10 == 0:
            print(f"Iteration {i}/{iterations}, Loss: {loss_value}")
 
    # Return the generated image
    return input_image[0]
 
# 5. Visualize the generated image that maximizes activation
def visualize_activation_image(activation_image):
    plt.imshow(activation_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title("Maximized Activation Image")
    plt.show()
 
# 6. Example usage
x_train, y_train, x_test, y_test = load_mnist_data()
 
# Build and train the CNN model
model = build_cnn_model()
model = train_model(model, x_train, y_train, x_test, y_test)
 
# Perform activation maximization on a specific layer
target_layer_name = "conv2d"  # You can try different layers like "conv2d" or "conv2d_1"
maximized_image = activation_maximization(model, target_layer_name, target_class=1, iterations=100, learning_rate=0.01)
 
# Visualize the image
visualize_activation_image(maximized_image)