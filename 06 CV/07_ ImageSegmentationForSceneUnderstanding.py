"""
Project 207. Image segmentation for scene understanding
Description:
This project involves segmenting an image into meaningful regions, helping AI understand scenes by classifying each pixel into categories like sky, road, person, car, etc. We’ll use a pre-trained semantic segmentation model from torchvision such as DeepLabV3 with a ResNet backbone. This model can distinguish different parts of an image and assign them to a specific class for enhanced scene comprehension.

About:
This implementation provides a visual breakdown of a scene by classifying each pixel, which is critical for tasks like autonomous driving, robotics, and image captioning.
"""

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
 
# Load a pre-trained DeepLabV3 model with a ResNet-101 backbone
# This model is trained on the COCO/Cityscapes dataset and supports semantic segmentation
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Set the model to evaluation mode
 
# Define a transformation to preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),           # Resize image to expected input size
    transforms.ToTensor(),                   # Convert image to PyTorch tensor
    transforms.Normalize(                    # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],          # Mean of RGB channels
        std=[0.229, 0.224, 0.225]            # Standard deviation of RGB channels
    )
])
 
# Load an image for segmentation
image_path = 'scene.jpg'  # Replace with your image path
input_image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB
input_tensor = preprocess(input_image)               # Apply transformations
input_batch = input_tensor.unsqueeze(0)              # Add batch dimension
 
# Move input to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_batch = input_batch.to(device)
 
# Perform inference to get segmentation output
with torch.no_grad():
    output = model(input_batch)['out'][0]  # Get output tensor for the first image in batch
 
# Apply argmax to get the class with highest probability for each pixel
segmentation = output.argmax(0).byte().cpu().numpy()  # Convert to NumPy array for visualization
 
# Define a colormap for visualization (Pascal VOC-style)
def decode_segmap(mask):
    label_colors = np.array([
        (0, 0, 0),         # background
        (128, 0, 0),       # aeroplane
        (0, 128, 0),       # bicycle
        (128, 128, 0),     # bird
        (0, 0, 128),       # boat
        (128, 0, 128),     # bottle
        (0, 128, 128),     # bus
        (128, 128, 128),   # car
        (64, 0, 0),        # cat
        (192, 0, 0),       # chair
        (64, 128, 0),      # cow
        (192, 128, 0),     # dining table
        (64, 0, 128),      # dog
        (192, 0, 128),     # horse
        (64, 128, 128),    # motorbike
        (192, 128, 128),   # person
        (0, 64, 0),        # potted plant
        (128, 64, 0),      # sheep
        (0, 192, 0),       # sofa
        (128, 192, 0),     # train
        (0, 64, 128)       # tv/monitor
    ])
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for label in range(len(label_colors)):
        idx = mask == label
        r[idx], g[idx], b[idx] = label_colors[label]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb
 
# Convert segmented mask to color map
seg_image = decode_segmap(segmentation)
 
# Display original and segmented image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(input_image)
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.title('Segmented Scene')
plt.imshow(seg_image)
plt.axis('off')
 
plt.tight_layout()
plt.show()