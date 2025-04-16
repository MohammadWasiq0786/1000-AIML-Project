"""
Project 5. Simple neural network from scratch
Description:
In this project, we implement a simple feedforward neural network from scratch in Python—without using any high-level frameworks like TensorFlow or PyTorch. This is a great way to understand the inner workings of neural networks, including forward propagation, loss calculation, and backpropagation using only NumPy.

About:
This neural network:

Has 2 input neurons, 1 hidden layer with 4 neurons, and 1 output neuron

Learns to model the XOR pattern (a nonlinear problem)

Is implemented using only NumPy—ideal for learning backpropagation math
"""

# Import required library
import numpy as np
 
# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def sigmoid_derivative(x):
    return x * (1 - x)
 
# Sample input data (4 examples, 2 features each)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
 
# Target outputs (XOR pattern for challenge)
y = np.array([[0], [1], [1], [0]])
 
# Set random seed for reproducibility
np.random.seed(42)
 
# Initialize weights and biases randomly
input_neurons = 2
hidden_neurons = 4
output_neurons = 1
 
# Weight matrix from input to hidden layer
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
# Bias for hidden layer
b1 = np.random.uniform(size=(1, hidden_neurons))
 
# Weight matrix from hidden to output layer
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
# Bias for output layer
b2 = np.random.uniform(size=(1, output_neurons))
 
# Training loop
epochs = 10000
learning_rate = 0.1
 
for epoch in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
 
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
 
    # Calculate the error (loss)
    error = y - final_output
 
    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
 
    # Update weights and biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
 
    # Print loss occasionally
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")
 
# Final results
print("\nFinal predictions after training:")
print(final_output.round(3))