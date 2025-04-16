"""
Project 209. Semantic segmentation implementation
Description:
Semantic segmentation classifies every pixel in an image into a predefined class (e.g., road, building, tree), without distinguishing between instances. It gives a global understanding of the scene layout. We'll use a pre-trained model like FCN (Fully Convolutional Network) with a ResNet backbone from torchvision to perform semantic segmentation.

About:
This implementation showcases how to label every pixel of an image with a category label, enabling machines to understand the layout and structure of environments—key for applications like autonomous navigation, urban planning, and smart surveillance.
"""

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 
# Load the pre-trained semantic segmentation model (FCN with ResNet-50 backbone)
model = models.segmentation.fcn_resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode
 
# Define transformation for input image
transform = transforms.Compose([
    transforms.Resize((520, 520)),           # Resize input to expected size
    transforms.ToTensor(),                   # Convert to PyTorch tensor
    transforms.Normalize(                    # Normalize using ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
 
# Load and preprocess the input image
image_path = 'cityscape.jpg'  # Replace with your image
input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
 
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)
 
# Run inference
with torch.no_grad():
    output = model(input_tensor)['out'][0]  # Get the segmentation map (shape: [num_classes, H, W])
 
# Get the predicted class for each pixel
segmentation_map = output.argmax(0).byte().cpu().numpy()
 
# Simple color palette (can be replaced with Cityscapes/VOC color maps)
def create_color_map(seg_map):
    # Define random but consistent colors for classes
    np.random.seed(10)
    num_classes = seg_map.max() + 1
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    
    h, w = seg_map.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label in range(num_classes):
        color_seg[seg_map == label] = colors[label]
    
    return color_seg
 
# Convert segmentation map to color image
colored_seg = create_color_map(segmentation_map)
 
# Display original image and segmented image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image)
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.title('Semantic Segmentation Output')
plt.imshow(colored_seg)
plt.axis('off')
 
plt.tight_layout()
plt.show()