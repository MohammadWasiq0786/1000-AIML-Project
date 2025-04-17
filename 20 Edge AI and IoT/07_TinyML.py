"""
Project 767: TinyML Implementations
Description
TinyML is the deployment of machine learning models on ultra-low-power devices like microcontrollers (e.g., Arduino, ESP32). These devices have limited CPU, memory, and power, so the models must be small and efficient. In this project, we'll train a small model on the MNIST dataset and convert it to a TensorFlow Lite for Microcontrollers-compatible format for deployment on embedded hardware.

This TFLite model can now be deployed using frameworks like TensorFlow Lite for Microcontrollers (TFLM) on platforms like Arduino Nano 33 BLE Sense or ESP32.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
 
# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
 
# Define a very small CNN suitable for microcontrollers
def tiny_model():
    model = models.Sequential([
        layers.Conv2D(8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model
 
# Create and train the model
model = tiny_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.1)
 
# Evaluate the model
loss, acc = model.evaluate(x_test, y_test)
print(f"✅ Tiny model accuracy: {acc:.4f}")
 
# Convert to TFLite (for microcontrollers)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
 
# Representative dataset for quantization
def representative_data_gen():
    for i in range(100):
        yield [x_train[i:i+1].astype(np.float32)]
 
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
 
# Save the quantized TFLite model
with open("mnist_tinyml_model.tflite", "wb") as f:
    f.write(tflite_model)
 
print("✅ TFLite model saved! Ready for deployment on microcontrollers using TinyML.")