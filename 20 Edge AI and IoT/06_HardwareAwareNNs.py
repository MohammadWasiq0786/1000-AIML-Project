"""
Project 766: Hardware-Aware Neural Networks
Description
Hardware-aware neural networks are designed by taking the hardware constraints—like memory, latency, and power consumption—into account during model development. This helps ensure the model performs optimally on edge devices such as microcontrollers or smartphones. We’ll simulate a hardware-constrained environment using TensorFlow Lite Model Maker with MobileNetV2 and set constraints like model size and latency during training.
This model uses a smaller width multiplier (0.35) and lower input resolution (96x96), making it lightweight and efficient on devices with limited compute.
"""

import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
import matplotlib.pyplot as plt
 
# Use TensorFlow’s built-in flower dataset for quick prototyping
import tensorflow_datasets as tfds
dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
 
# Split into train and test
train_data = dataset['train'].take(3000)
test_data = dataset['train'].skip(3000)
 
# Save to directory (Model Maker expects image folders, this step is typically needed if not using built-in sets)
 
# Load dataset using Model Maker’s DataLoader
data = DataLoader.from_tensorflow_datasets('tf_flowers', shuffle=True, split=[0.8, 0.2])
 
# Create a model using MobileNetV2 with reduced input size to fit edge constraints
model = image_classifier.create(
    data[0],  # training data
    model_spec=image_classifier.ModelSpec(
        uri='mobilenet_v2_035_96',  # 0.35 width multiplier, 96x96 input resolution
    ),
    epochs=3,
    batch_size=32
)
 
# Evaluate the model on the test set
loss, acc = model.evaluate(data[1])
print(f"✅ Model accuracy: {acc:.4f} on hardware-constrained configuration")
 
# Export to TensorFlow Lite model
model.export(export_dir='hardware_aware_model')
 
print("✅ Hardware-aware MobileNetV2 model saved and ready for edge deployment.")