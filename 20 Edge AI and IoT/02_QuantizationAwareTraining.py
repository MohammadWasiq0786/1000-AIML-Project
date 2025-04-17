"""
Project 762: Quantization-Aware Training
Description
Quantization-aware training (QAT) is a technique to train neural networks while simulating low-precision arithmetic (like int8) during training. Unlike post-training quantization, QAT allows the model to adapt to the quantization effects, preserving higher accuracy. It’s ideal for deploying deep learning models to edge devices with strict resource constraints.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.datasets import mnist
 
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Normalize and reshape data
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]
x_test  = x_test[..., tf.newaxis]
 
# Build a simple CNN model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model
 
# Create the base model
model = build_model()
 
# Apply quantization-aware training using the TensorFlow Model Optimization Toolkit
quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(model)
 
# Compile the quantized model
qat_model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
 
# Train the model with QAT
qat_model.fit(x_train, y_train, epochs=3, validation_split=0.1)
 
# Evaluate the QAT model on the test set
loss, accuracy = qat_model.evaluate(x_test, y_test)
print(f"✅ Quantization-aware trained model accuracy: {accuracy:.4f}")
 
# Convert to a TensorFlow Lite model with int8 weights
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_qat_model = converter.convert()
 
# Save the final TFLite model
with open("mnist_qat_model.tflite", "wb") as f:
    f.write(tflite_qat_model)
 
print("✅ Quantization-aware trained TFLite model saved! Ideal for edge deployment.")