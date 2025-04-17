"""
Project 735: Adversarial Example Generation
Description:
Adversarial examples are inputs designed to fool machine learning models by introducing small, imperceptible perturbations that cause the model to make incorrect predictions. These examples are particularly dangerous for deep learning models and can be used to test their robustness. In this project, we will generate adversarial examples for a trained model using the Fast Gradient Sign Method (FGSM), which perturbs the input image in the direction of the gradient of the loss with respect to the input image. This technique is commonly used to generate adversarial examples for image classification models.

Explanation:
Image Preprocessing: The load_image() function loads and preprocesses an image, resizing it to 224x224 pixels and normalizing the pixel values for VGG16.

Model Loading: We load the pre-trained VGG16 model from Keras, which is trained on ImageNet and can be used for image classification tasks.

Adversarial Example Generation (FGSM):

The generate_adversarial_example() function calculates the gradient of the loss with respect to the input image. This gradient represents how the model’s output would change if we perturbed each pixel of the image.

The Fast Gradient Sign Method (FGSM) generates the perturbation by taking the sign of the gradient and multiplying it by a small constant epsilon.

The perturbed image is then created by adding the perturbation to the original image, and the result is clipped to ensure the pixel values remain valid.

Visualization: The visualize_adversarial_examples() function displays both the original and adversarial images side by side, allowing us to visually inspect the impact of the adversarial perturbations.

Adversarial examples can be used to evaluate the robustness of a model by testing how vulnerable it is to small perturbations that cause incorrect predictions. In practice, adversarial attacks are used to identify weaknesses in machine learning models, and adversarial training can help make models more resilient.
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
 
# 4. Visualize the original and adversarial images
def visualize_adversarial_examples(original_image, adversarial_image):
    """
    Visualize the original and adversarial images side by side.
    """
    plt.figure(figsize=(12, 6))
 
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0] / 255.0)
    plt.title("Original Image")
    plt.axis('off')
 
    # Adversarial image
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image[0] / 255.0)
    plt.title("Adversarial Image")
    plt.axis('off')
 
    plt.show()
 
# 5. Example usage
img_path = 'path_to_image.jpg'  # Replace with the image path
img_array = load_image(img_path)
 
# Load the pre-trained model
model = load_vgg16_model()
 
# Generate the adversarial example
adversarial_image = generate_adversarial_example(model, img_array, epsilon=0.1)
 
# Visualize the original and adversarial images
visualize_adversarial_examples(img_array, adversarial_image)
