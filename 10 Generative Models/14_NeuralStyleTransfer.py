"""
Project 374. Neural style transfer
Description:
Neural Style Transfer (NST) is a technique that blends the content of one image with the style of another using a pre-trained convolutional neural network (CNN). This process allows the generation of images that retain the content of the original image while adopting the artistic style of another image. It’s widely used for creating artistic renderings of photos.

In this project, we’ll implement Neural Style Transfer to transform an image into the style of a famous painting or artwork, while maintaining the content of the original image.

About:
✅ What It Does:
Content Loss ensures the content of the generated image matches the content of the original image

Style Loss ensures the generated image adopts the artistic style of the reference image by matching the Gram matrix of activations in selected layers of the VGG19 network

Uses the pre-trained VGG19 network to extract features from both the content and style images

LBFGS optimizer is used to iteratively adjust the generated image, minimizing the total loss (content + style)

The generated image is displayed every 50 epochs during the training process
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
 
# 1. Load and preprocess the images
def load_image(image_path, max_size=400):
    image = Image.open(image_path)
    # Scale the image to the maximum size while maintaining aspect ratio
    scale = max_size / float(max(image.size))
    new_size = tuple([int(dim * scale) for dim in image.size])
    image = image.resize(new_size, Image.ANTIALIAS)
    
    # Apply the necessary transformations to the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image
 
# 2. Define the loss functions (content and style)
def content_loss(content, target):
    return torch.mean((content - target)**2)
 
def style_loss(style, target):
    # Compute the Gram matrix of the style image
    gram_style = torch.mm(style.view(style.size(1), -1), style.view(style.size(1), -1).t())
    gram_target = torch.mm(target.view(target.size(1), -1), target.view(target.size(1), -1).t())
    return torch.mean((gram_style - gram_target)**2)
 
# 3. Load the pre-trained VGG19 model (for feature extraction)
vgg = models.vgg19(pretrained=True).features.eval()
 
# 4. Define the model for Neural Style Transfer
class NSTModel(nn.Module):
    def __init__(self):
        super(NSTModel, self).__init__()
        self.vgg = vgg
        self.content_layers = ['4']  # Layer to extract content
        self.style_layers = ['1', '6', '11', '20']  # Layers to extract style
 
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if str(i) in self.content_layers:
                features.append(x)
            elif str(i) in self.style_layers:
                features.append(x)
        return features
 
# 5. Set up the model and optimizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NSTModel().to(device)
 
# Load content and style images
content_image = load_image('path_to_your_content_image.jpg').to(device)
style_image = load_image('path_to_your_style_image.jpg').to(device)
 
# Use the same image as the starting point for optimization
target_image = content_image.clone().requires_grad_(True)
 
optimizer = optim.LBFGS([target_image])
 
# 6. Training loop for Neural Style Transfer
num_epochs = 500
for epoch in range(num_epochs):
    def closure():
        target_image.data.clamp_(0, 1)  # Clamp the values to [0, 1]
        optimizer.zero_grad()
 
        # Extract features from content, style, and target images
        content_features = model(content_image)
        style_features = model(style_image)
        target_features = model(target_image)
 
        # Compute content and style losses
        content_loss_value = content_loss(content_features[0], target_features[0])
        style_loss_value = 0.0
        for style, target in zip(style_features, target_features[1:]):
            style_loss_value += style_loss(style, target)
        
        total_loss = content_loss_value + 1e6 * style_loss_value  # Combine losses with a weight for style
        total_loss.backward()
 
        return total_loss
 
    optimizer.step(closure)
 
    # Display image every few epochs
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item()}')
        plt.imshow(target_image.cpu().clone().detach().squeeze(0).permute(1, 2, 0).numpy())
        plt.show()