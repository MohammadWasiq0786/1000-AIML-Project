"""
Project 208. Instance segmentation implementation
Description:
Unlike semantic segmentation (which labels pixels based on class), instance segmentation identifies each individual object in an image, even if they belong to the same class (e.g., two people = two instances of the "person" class). We'll use a pre-trained Mask R-CNN model from torchvision, which returns bounding boxes, class labels, and pixel-wise masks for each detected object.

About:
This project allows AI to not just detect objects but differentiate each individual instance — a crucial skill for robots, self-driving cars, and scene parsing in AR/VR systems.
"""

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  # For drawing and masking
 
# Load the pre-trained Mask R-CNN model with a ResNet-50 backbone
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode
 
# Define image transforms to match model input requirements
transform = transforms.Compose([
    transforms.ToTensor()  # Converts PIL image to PyTorch tensor (normalizes 0-1)
])
 
# Load an image and convert to RGB
image_path = 'objects.jpg'  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
 
# Use CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_tensor = image_tensor.to(device)
 
# Perform inference
with torch.no_grad():
    predictions = model(image_tensor)[0]  # Get prediction dictionary for the image
 
# Set confidence threshold to filter out weak detections
confidence_threshold = 0.7
 
# Extract data from predictions
boxes = predictions['boxes'].cpu().numpy()
labels = predictions['labels'].cpu().numpy()
scores = predictions['scores'].cpu().numpy()
masks = predictions['masks'].cpu().numpy()  # Each mask has shape (1, H, W)
 
# Create a blank image for visualization
image_np = np.array(image)
output_image = image_np.copy()
 
# Colors for different instances
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(boxes), 3), dtype=np.uint8)
 
# Draw each detected object
for i in range(len(boxes)):
    if scores[i] < confidence_threshold:
        continue
 
    # Get the mask and apply threshold to make it binary
    mask = masks[i, 0]
    binary_mask = mask > 0.5
 
    # Create a colored mask
    color_mask = colors[i]
    colored_region = np.zeros_like(image_np)
    for c in range(3):  # RGB channels
        colored_region[:, :, c] = binary_mask * color_mask[c]
 
    # Blend original image with mask (alpha blending)
    output_image = cv2.addWeighted(output_image, 1.0, colored_region, 0.5, 0)
 
    # Draw bounding box
    x1, y1, x2, y2 = boxes[i].astype(int)
    cv2.rectangle(output_image, (x1, y1), (x2, y2), tuple(color_mask.tolist()), 2)
 
    # Label object
    label = labels[i]
    cv2.putText(output_image, f"ID {label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(color_mask.tolist()), 2)
 
# Show the image with instance masks
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title('Instance Segmentation Result')
plt.axis('off')
plt.show()