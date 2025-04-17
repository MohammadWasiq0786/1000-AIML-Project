"""
Project 764: Neural Architecture Search for Edge
Description
Neural Architecture Search (NAS) automates the design of neural networks. When targeting edge devices, the search must balance accuracy with constraints like latency, memory, and energy consumption. In this project, we’ll use Keras Tuner to perform a simple NAS to find an optimal model architecture for MNIST that fits within edge deployment limits.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.datasets import mnist
 
# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
 
# Define a model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
 
    # Tune number of convolutional layers: 1 to 3
    for i in range(hp.Int("conv_layers", 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Int(f"filters_{i}", min_value=16, max_value=64, step=16),
            kernel_size=hp.Choice(f"kernel_size_{i}", values=[3, 5]),
            activation="relu"
        ))
        model.add(layers.MaxPooling2D(pool_size=2))
 
    model.add(layers.Flatten())
 
    # Tune number of dense units
    model.add(layers.Dense(
        units=hp.Int("dense_units", 32, 128, step=32),
        activation="relu"
    ))
    model.add(layers.Dense(10))  # Output layer
 
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model
 
# Initialize Keras Tuner
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="nas_edge_mnist",
    project_name="edge_nas_demo"
)
 
# Perform NAS to find best architecture
tuner.search(x_train, y_train, epochs=3, validation_split=0.1)
 
# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
 
# Evaluate the best model
loss, acc = best_model.evaluate(x_test, y_test)
print(f"✅ Best model accuracy: {acc:.4f}")
 
# Save for edge deployment
best_model.save("best_edge_model.h5")
print("✅ NAS-based model saved and ready for compression or deployment.")