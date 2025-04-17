"""
Project 799: Low-Power Computer Vision
Description
Low-power computer vision focuses on enabling image-based AI tasks like classification or detection on energy-constrained devices (e.g., microcontrollers, IoT cams). This project demonstrates how to build and compress a tiny image classifier (e.g., for plant health, trash sorting, etc.) using MobileNetV2 + TensorFlow Lite quantization for deployment on devices like ESP32-CAM or Raspberry Pi Zero.

This project is deployable on devices like:

ESP32-CAM with TFLite Micro

Raspberry Pi Zero (Lite)

Arduino Portenta H7 + vision shield
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
 
# Load small image dataset (CIFAR-10: 32x32 color images, 10 classes)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
 
# Keep only 3 classes for simplification (e.g., airplane, car, bird)
selected_classes = [0, 1, 2]
train_mask = np.isin(y_train, selected_classes).flatten()
test_mask = np.isin(y_test, selected_classes).flatten()
 
x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]
 
# Normalize labels to 0–2
label_map = {k: i for i, k in enumerate(selected_classes)}
y_train = np.vectorize(label_map.get)(y_train)
y_test = np.vectorize(label_map.get)(y_test)
 
# Build a tiny CNN model suitable for edge deployment
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)
 
# Evaluate accuracy
loss, acc = model.evaluate(x_test, y_test)
print(f"✅ Low-Power Vision Model Accuracy: {acc:.4f}")
 
# Convert to TFLite with full quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
 
def representative_data_gen():
    for input_value in x_train[:100]:
        yield [np.expand_dims(input_value, axis=0)]
 
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
 
tflite_model = converter.convert()
 
# Save quantized model
with open("tiny_cnn_quantized.tflite", "wb") as f:
    f.write(tflite_model)
 
print("✅ Quantized Tiny Image Classifier saved — ready for low-power deployment!")