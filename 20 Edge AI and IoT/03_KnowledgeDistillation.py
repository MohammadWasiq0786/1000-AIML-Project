"""
Project 763: Knowledge Distillation Implementation
Description
Knowledge Distillation is a model compression technique where a smaller model (called the student) is trained to mimic a larger, well-performing model (called the teacher). Instead of training on the hard labels alone, the student learns from the soft probabilities predicted by the teacher, which carry more information. This enables high accuracy in smaller models suitable for edge devices.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
 
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
 
# Define a large model to act as the teacher
def create_teacher():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    return model
 
# Define a smaller student model
def create_student():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    return model
 
# Compile and train the teacher model
teacher = create_teacher()
teacher.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
teacher.fit(x_train, y_train, epochs=3, validation_split=0.1)
 
# Get soft targets from the teacher model (soft labels = logits)
teacher_logits = tf.nn.softmax(teacher.predict(x_train) / 5.0)  # temperature = 5
 
# Define a custom distillation loss function
def distillation_loss(y_true, y_pred, teacher_soft, alpha=0.5, temperature=5.0):
    # Hard loss: ground-truth labels
    hard_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
 
    # Soft loss: teacher's soft probabilities vs student output
    y_soft = tf.nn.softmax(y_pred / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(teacher_soft, y_soft)
 
    return alpha * hard_loss + (1 - alpha) * soft_loss
 
# Custom training loop for distillation
student = create_student()
optimizer = tf.keras.optimizers.Adam()
 
# Convert teacher soft labels to tensor
teacher_soft_labels = tf.convert_to_tensor(teacher_logits)
 
# Prepare the training dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train, teacher_soft_labels)).batch(64)
 
# Training student model using distillation
for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    for batch_x, batch_y, batch_soft in train_ds:
        with tf.GradientTape() as tape:
            predictions = student(batch_x, training=True)
            loss = distillation_loss(batch_y, predictions, batch_soft)
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
    print(f"✅ Epoch {epoch + 1} complete.")
 
# Evaluate student model
student.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
student.evaluate(x_test, y_test)
 
print("✅ Knowledge distillation complete. Student model is compact and edge-ready.")