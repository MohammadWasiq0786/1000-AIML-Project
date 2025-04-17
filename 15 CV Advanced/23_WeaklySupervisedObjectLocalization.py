"""
Project 583: Weakly Supervised Object Localization
Description:
Weakly supervised object localization involves identifying the locations of objects within images without using full bounding box annotations. This task relies on class labels or image-level annotations to localize the objects. In this project, we will use a pre-trained model to perform weakly supervised localization of objects in images.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
 
# 1. Load pre-trained ResNet model for object localization
model = models.resnet50(pretrained=True)
model.eval()
 
# 2. Load and preprocess the image
image_path = "path_to_image.jpg"  # Replace with an actual image path
image = Image.open(image_path)
 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
 
# 3. Get the class label prediction
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted_class = torch.max(outputs, 1)
 
# 4. Get the gradients for the class output
class_idx = predicted_class.item()
model.zero_grad()
one_hot_output = torch.FloatTensor(1, outputs.size()[-1]).zero_()
one_hot_output[0][class_idx] = 1
outputs.backward(gradient=one_hot_output)
 
# 5. Generate the heatmap (using Grad-CAM)
grads = model.layer4[2].conv3.weight.grad
target = model.layer4[2].conv3.output
weights = grads.mean(dim=[0, 2, 3])  # Average gradients across the spatial dimensions
activation_map = torch.sum(weights * target, dim=1).squeeze().detach().cpu().numpy()
 
# 6. Visualize the heatmap
activation_map = cv2.resize(activation_map, (image.size[0], image.size[1]))
activation_map = np.maximum(activation_map, 0)
activation_map = activation_map / activation_map.max()  # Normalize to [0, 1]
 
# 7. Display the image with the heatmap overlaid
plt.imshow(image)
plt.imshow(activation_map, alpha=0.5, cmap='jet')  # Overlay the heatmap with transparency
plt.title(f"Predicted Class: {imagenet_labels[predicted_class.item()]}")
plt.show()