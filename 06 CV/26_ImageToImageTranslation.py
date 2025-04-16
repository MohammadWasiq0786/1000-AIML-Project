"""
Project 226. Image-to-image translation
Description:
Image-to-image translation is the task of transforming one type of image into another while preserving the structure—e.g., sketch to photo, day to night, summer to winter, or black-and-white to color. In this project, we'll use a pre-trained Pix2Pix model (a conditional GAN) from the torchvision or fastai ecosystem to perform sketch-to-image translation.

About:
What It Does:
This project performs controlled transformation of images using learned visual mappings. Applications include photo enhancement, artistic style transfer, object reconstruction, and even medical image synthesis. You can fine-tune your own Pix2Pix model or use datasets like Facades, Maps, or Edges2Shoes for specific tasks.
"""
# Install required packages:
# pip install torch torchvision matplotlib transformers diffusers accelerate
 
import torch
from diffusers import StableDiffusionPix2PixPipeline
from PIL import Image
import matplotlib.pyplot as plt
 
# Load a pre-trained Pix2Pix pipeline (based on Stable Diffusion finetuned for image-to-image translation)
pipe = StableDiffusionPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to("cuda")
 
# Load your input image (e.g., sketch or photo)
input_image = Image.open("sketch_sample.jpg").convert("RGB")  # Replace with your input
 
# Define instruction (this is a prompt-based conditional translation)
prompt = "Convert this sketch into a realistic photo"
 
# Generate translated image
output = pipe(prompt=prompt, image=input_image, num_inference_steps=50).images[0]
 
# Display original and translated image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Input Image")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("Translated Output")
plt.axis('off')
 
plt.tight_layout()
plt.show()