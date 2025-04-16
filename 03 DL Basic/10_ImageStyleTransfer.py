"""
Project 90. Image style transfer
Description:
Image style transfer is the process of combining the content of one image with the style of another using a deep neural network. This project implements a basic Neural Style Transfer using a pre-trained VGG19 model from Keras to create stylized images.

About:
🖼️ What This Project Demonstrates:
Transfers artistic style from one image to another

Uses pre-trained VGG19 for feature extraction

Balances content loss and style loss to create stylized output
"""

# Install if not already: pip install tensorflow matplotlib pillow
 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.image import resize
 
# Function to load and preprocess image
def load_and_process_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)
 
# Function to deprocess image for display
def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    return np.clip(x[:, :, ::-1], 0, 255).astype('uint8')
 
# Load images
content_path = tf.keras.utils.get_file("content.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_path = tf.keras.utils.get_file("style.jpg", "https://i.imgur.com/YoNe9Hk.jpg")
 
content_image = load_and_process_image(content_path)
style_image = load_and_process_image(style_path)
 
# Define content and style layers
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
 
# Load VGG19 model for feature extraction
vgg = vgg19.VGG19(weights='imagenet', include_top=False)
vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]
model = Model([vgg.input], outputs)
 
# Compute feature representations
def get_features(image):
    preprocessed = tf.convert_to_tensor(image, dtype=tf.float32)
    return model(preprocessed)
 
style_features = get_features(style_image)
content_features = get_features(content_image)
 
# Extract style and content features separately
style_acts = style_features[:-1]
content_act = content_features[-1]
 
# Gram matrix for style
def gram_matrix(tensor):
    x = tf.squeeze(tensor)
    x = tf.reshape(x, (-1, x.shape[-1]))
    return tf.matmul(x, x, transpose_a=True)
 
style_grams = [gram_matrix(f) for f in style_acts]
 
# Initialize generated image
generated_image = tf.Variable(content_image, dtype=tf.float32)
 
# Define loss and optimizer
optimizer = tf.optimizers.Adam(learning_rate=0.02)
 
@tf.function
def compute_loss_and_grads(generated_image):
    with tf.GradientTape() as tape:
        features = get_features(generated_image)
        gen_style = features[:-1]
        gen_content = features[-1]
 
        style_loss = tf.add_n([
            tf.reduce_mean((gram_matrix(gen) - style_gram) ** 2)
            for gen, style_gram in zip(gen_style, style_grams)
        ]) / len(style_layers)
 
        content_loss = tf.reduce_mean((gen_content - content_act) ** 2)
 
        total_loss = 1e4 * style_loss + content_loss
    grads = tape.gradient(total_loss, generated_image)
    return total_loss, grads
 
# Run optimization loop
epochs = 500
for i in range(epochs):
    loss, grads = compute_loss_and_grads(generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.numpy():.2f}")
 
# Show final output
final_img = deprocess_image(generated_image.numpy())
plt.imshow(final_img)
plt.title("Stylized Image")
plt.axis("off")
plt.tight_layout()
plt.show()