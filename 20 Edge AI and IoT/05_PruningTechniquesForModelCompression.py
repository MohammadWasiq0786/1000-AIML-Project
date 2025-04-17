"""
Project 765: Pruning Techniques for Model Compression
Description
Pruning is a model compression technique that removes less significant weights from the network, effectively "sparsifying" the model. This reduces storage size and computation, making it ideal for deployment on edge devices. TensorFlow's Model Optimization Toolkit allows structured or unstructured pruning during or after training.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
 
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]
x_test  = x_test[..., tf.newaxis]
 
# Define a simple CNN model
def create_model():
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(10)
    ])
 
# Create the base model
model = create_model()
 
# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,        # 50% of weights will be zero
        begin_step=0,
        end_step=np.ceil(len(x_train) / 128).astype(np.int32) * 3  # 3 epochs
    )
}
 
# Apply pruning wrapper to the model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
 
# Compile the pruned model
pruned_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
 
# Define pruning callbacks
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
 
# Train the pruned model
pruned_model.fit(x_train, y_train, batch_size=128, epochs=3,
                 validation_split=0.1, callbacks=callbacks)
 
# Strip pruning wrappers for deployment
final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
 
# Save the pruned model
final_model.save("pruned_mnist_model.h5")
print("✅ Pruned model saved! Reduced size and computation for edge use.")