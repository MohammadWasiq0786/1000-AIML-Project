"""
Project 736: Adversarial Robustness Evaluation
Description:
Adversarial robustness evaluation is the process of assessing how resistant a machine learning model is to adversarial attacks—small, carefully crafted perturbations in the input data that can cause the model to make incorrect predictions. By generating adversarial examples (such as with FGSM or PGD) and evaluating the model's performance on these examples, we can determine how vulnerable the model is to adversarial inputs. In this project, we will generate adversarial examples and evaluate the accuracy and robustness of a model under attack.

Explanation:
Image Preprocessing: The load_image() function loads and preprocesses the input image, resizing it to 224x224 pixels and normalizing the pixel values for VGG16.

Model Loading: We load the pre-trained VGG16 model from Keras, which is trained on ImageNet.

Adversarial Example Generation (FGSM): The generate_adversarial_example() function generates adversarial examples using the Fast Gradient Sign Method (FGSM). The perturbation is applied to the input image in the direction of the gradient of the loss function with respect to the input image.

Robustness Evaluation: The evaluate_adversarial_robustness() function evaluates the model's robustness by making predictions on both the original and adversarial images. It compares the model's predictions and checks if the model correctly classifies both the original and adversarial images.

Output: The output shows the model's performance on both the original and adversarial examples, indicating whether the model was fooled by the adversarial perturbations.

This project is useful for assessing the adversarial robustness of models, helping us understand their vulnerability to adversarial attacks.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
 
# 1. Load and preprocess the image
def load_image(img_path):
    """
    Load an image from the given path and preprocess it for VGG16.
    - Resizes the image to 224x224, converts to array, and normalizes it.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # VGG16 expects 224x224 images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Preprocess for VGG16
    return img_array
 
# 2. Load pre-trained VGG16 model
def load_vgg16_model():
    """
    Load the pre-trained VGG16 model with ImageNet weights.
    """
    model = VGG16(weights='imagenet')
    return model
 
# 3. Generate adversarial example using Fast Gradient Sign Method (FGSM)
def generate_adversarial_example(model, img_array, epsilon=0.1):
    """
    Generate adversarial example using the Fast Gradient Sign Method (FGSM).
    - Perturbs the input image in the direction of the gradient of the loss.
    """
    # Set the input image as a TensorFlow variable
    img_tensor = tf.Variable(img_array)
 
    # Compute the gradient of the loss with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(tf.argmax(predictions, axis=1), predictions)
    
    # Get the gradient of the loss with respect to the input image
    gradient = tape.gradient(loss, img_tensor)
 
    # Compute the perturbation using the sign of the gradient
    perturbation = epsilon * tf.sign(gradient)
    adversarial_image = img_tensor + perturbation  # Apply the perturbation
 
    # Clip the pixel values to keep the image valid
    adversarial_image = tf.clip_by_value(adversarial_image, -1.0, 1.0)
 
    return adversarial_image.numpy()
 
# 4. Evaluate model on adversarial examples
def evaluate_adversarial_robustness(model, img_array, true_class, epsilon=0.1):
    """
    Evaluate the model's robustness by checking its performance on both original and adversarial examples.
    """
    # Generate adversarial example
    adversarial_image = generate_adversarial_example(model, img_array, epsilon)
 
    # Predict on original and adversarial images
    original_prediction = np.argmax(model.predict(img_array))
    adversarial_prediction = np.argmax(model.predict(adversarial_image))
 
    # Compare predictions
    print(f"Original Prediction: {original_prediction}, True Class: {true_class}")
    print(f"Adversarial Prediction: {adversarial_prediction}, True Class: {true_class}")
    
    if original_prediction == true_class:
        print("Model correctly classified the original image.")
    else:
        print("Model misclassified the original image.")
        
    if adversarial_prediction == true_class:
        print("Model correctly classified the adversarial image.")
    else:
        print("Model misclassified the adversarial image.")
 
# 5. Example usage
img_path = 'path_to_image.jpg'  # Replace with the image path
img_array = load_image(img_path)
 
# Load the pre-trained model
model = load_vgg16_model()
 
# Get the true class of the image
true_class = 5  # Example: Replace with the true class index of the image
 
# Evaluate the model's adversarial robustness
evaluate_adversarial_robustness(model, img_array, true_class, epsilon=0.1)