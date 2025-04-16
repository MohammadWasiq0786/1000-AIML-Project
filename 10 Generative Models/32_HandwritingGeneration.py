"""
Project 392. Handwriting generation
Description:
Handwriting generation is the task of generating images of handwritten text based on a given input text. This is useful in applications like document generation, signature creation, and customized handwritten notes. Models like GANs, VAEs, and Recurrent Neural Networks (RNNs) can be used for this task, learning the characteristics of handwriting and generating realistic handwritten text. In this project, we will use a pre-trained model to generate handwritten text from an input string.

About:
✅ What It Does:
Generates an image of handwritten text based on the input string.

Uses a GAN-based model to learn handwriting patterns and generate realistic handwriting images.

Displays the generated handwritten text as an image.

Key features:
Handwriting Generation leverages a pre-trained model to generate images of text in a handwritten style.

The generated image can be used in applications such as document generation, personalized notes, or creative design.
"""

from handwriting_synthesis import HandwritingGenerator
import matplotlib.pyplot as plt
 
# 1. Load the pre-trained Handwriting Generation model
generator = HandwritingGenerator()
 
# 2. Define the input text you want to generate handwriting for
input_text = "Hello, this is a handwritten text generation example!"
 
# 3. Generate handwritten text image
handwritten_image = generator.generate(input_text)
 
# 4. Display the generated handwriting
plt.imshow(handwritten_image, cmap='gray')
plt.axis('off')  # Hide axes for better visualization
plt.title(f"Handwritten Text: {input_text}")
plt.show()