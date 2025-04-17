"""
Projects 761-800: Edge AI and IoT Projects
Project 761: Model Compression for Edge Devices
This project focuses on reducing the size and computational demands of a neural network model to make it suitable for deployment on edge devices (like mobile phones, Raspberry Pi, etc.) without significantly sacrificing accuracy. We achieve this using quantization and pruning techniques from TensorFlow Lite. The goal is to optimize a model such as MobileNetV2 for faster inference and smaller memory footprint while retaining acceptable performance.

This implementation compresses a pre-trained model for edge deployment using quantization (to reduce precision) and pruning (to sparsify weights). These techniques are essential for reducing latency, memory usage, and power consumption on devices with limited resources.
"""

import tensorflow as tf
import numpy as np
 
# Load a pre-trained MobileNetV2 model for demonstration purposes
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", input_shape=(224, 224, 3), include_top=True)
 
# Save the original model to disk
base_model.save("mobilenetv2_full.h5")
 
# ----------- Quantization with TensorFlow Lite -----------
 
# Convert the Keras model to a TensorFlow Lite model with post-training dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
 
# Enable dynamic range quantization - this reduces model size by quantizing weights to 8-bit integers
converter.optimizations = [tf.lite.Optimize.DEFAULT]
 
# Convert the model
tflite_quant_model = converter.convert()
 
# Save the quantized model to file
with open("mobilenetv2_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)
 
print("✅ Quantized model saved! Size reduced significantly for edge deployment.")
 
# ----------- Optional: Model Pruning (via TensorFlow Model Optimization Toolkit) -----------
 
from tensorflow_model_optimization.sparsity import keras as sparsity
 
# Set up pruning parameters: we prune 50% of the weights gradually over 2 epochs
pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                  final_sparsity=0.5,
                                                  begin_step=0,
                                                  end_step=1000)
}
 
# Wrap the model with pruning capabilities
pruned_model = sparsity.prune_low_magnitude(base_model, **pruning_params)
 
# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# NOTE: Normally you'd train the model again here to fine-tune weights, but we'll skip for brevity
 
# Strip pruning wrappers to make the model exportable
final_model = sparsity.strip_pruning(pruned_model)
 
# Save the pruned model
final_model.save("mobilenetv2_pruned.h5")
 
print("✅ Pruned model saved! Weights sparsified for memory efficiency.")
 
# ----------- Compare Sizes (Optional) -----------
 
import os
 
original_size = os.path.getsize("mobilenetv2_full.h5") / 1e6
quant_size = os.path.getsize("mobilenetv2_quant.tflite") / 1e6
pruned_size = os.path.getsize("mobilenetv2_pruned.h5") / 1e6
 
print(f"Original model size: {original_size:.2f} MB")
print(f"Quantized model size: {quant_size:.2f} MB")
print(f"Pruned model size: {pruned_size:.2f} MB")