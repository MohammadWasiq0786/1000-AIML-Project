"""
Project 138. DeepDream implementation
Description:
DeepDream is a computer vision technique that uses a trained convolutional neural network (CNN) to enhance and amplify patterns in input images. By maximizing the activation of certain layers, DeepDream hallucinates features like eyes, swirls, and structures, creating surreal dream-like visuals. In this project, we implement DeepDream using a pre-trained CNN like InceptionV3.

About:
🧠 What This Project Demonstrates:
Uses a pre-trained CNN (InceptionV3) to create surreal visuals

Implements gradient ascent on input image to enhance neuron activations

Captures the internal dreams of a deep network
"""

# Install if not already: pip install torch torchvision pillow matplotlib
 
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 
# Load pre-trained model
model = models.inception_v3(pretrained=True, aux_logits=False)
model.eval()
 
# Choose a specific layer to enhance
selected_layer = model.Mixed_5b  # Can try Mixed_6a, etc.
 
# Hook to capture activations
activations = None
def hook_fn(module, input, output):
    global activations
    activations = output
 
selected_layer.register_forward_hook(hook_fn)
 
# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])
 
def deprocess(img_tensor):
    img = img_tensor.clone().detach()
    img = img.squeeze().permute(1, 2, 0).numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
    img = np.clip(img, 0, 1)
    return img
 
# Load image
image_path = "sample.jpg"  # Replace with your image
image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image).unsqueeze(0).requires_grad_(True)
 
# DeepDream optimization
optimizer = torch.optim.Adam([input_tensor], lr=0.01)
 
for i in range(30):  # Number of iterations
    optimizer.zero_grad()
    model(input_tensor)
    loss = activations.norm()  # Maximize activations
    loss.backward()
    optimizer.step()
 
    if i % 10 == 0:
        print(f"🌀 Iteration {i}, Loss: {loss.item():.4f}")
 
# Show DeepDream result
dream_image = deprocess(input_tensor)
plt.imshow(dream_image)
plt.axis("off")
plt.title("💭 DeepDream Image")
plt.show()