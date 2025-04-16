"""
Project 124. CycleGAN for image-to-image translation
Description:
CycleGAN is a deep learning framework that learns to perform image-to-image translation without paired examples. For example, it can convert images from one domain (e.g., horses) to another (e.g., zebras), or photos to paintings. In this project, we use a pre-trained CycleGAN model from TensorFlow or PyTorch to translate images between domains.

About:
🧠 What This Project Demonstrates:
Performs unpaired image-to-image translation using CycleGAN

Translates styles between domains like horses ↔ zebras, summer ↔ winter, etc.

Uses pre-trained models for quick experimentation
"""

# Install if not already: pip install torch torchvision matplotlib pillow
 
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
 
# Load a pre-trained CycleGAN model from Torch Hub
model = torch.hub.load('junyanz/pytorch-CycleGAN-and-pix2pix', 'horse2zebra', pretrained=True)
model.eval()
 
# Load and preprocess input image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Shape: (1, 3, 256, 256)
 
# Load image of a horse (replace with your own)
input_image = load_image("horse.jpg")
with torch.no_grad():
    output_image = model(input_image)[0]
 
# Convert output tensor to image
def tensor_to_image(tensor):
    image = tensor.cpu().clone().detach()
    image = image.numpy().transpose(1, 2, 0)
    image = (image + 1) / 2.0  # Un-normalize
    return image
 
# Display results
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(tensor_to_image(input_image[0]))
plt.title("Original Image (Horse)")
plt.axis("off")
 
plt.subplot(1, 2, 2)
plt.imshow(tensor_to_image(output_image))
plt.title("Translated Image (Zebra)")
plt.axis("off")
 
plt.tight_layout()
plt.show()