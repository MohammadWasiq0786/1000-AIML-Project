"""
Project 582: Fine-grained Visual Categorization
Description:
Fine-grained visual categorization focuses on distinguishing subtle differences between objects of the same category, such as identifying different breeds of dogs or distinguishing between types of flowers. In this project, we will use a pre-trained model to perform fine-grained classification, leveraging the model's ability to learn detailed visual features.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
 
# 1. Load pre-trained ResNet model for fine-grained categorization
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode
 
# 2. Load and preprocess the image for fine-grained classification
image_path = "path_to_image.jpg"  # Replace with an actual image path
image = Image.open(image_path)
 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
 
# 3. Perform forward pass through the model
with torch.no_grad():
    outputs = model(input_tensor)
 
# 4. Get the predicted class label
_, predicted_class = torch.max(outputs, 1)
 
# 5. Load ImageNet labels (for illustration, here we use the ImageNet labels)
imagenet_labels = [line.strip() for line in open("imagenet_class_index.json")]
 
# 6. Display the predicted label
predicted_label = imagenet_labels[predicted_class.item()]
print(f"Predicted fine-grained label: {predicted_label}")
 
# 7. Display the image for reference
plt.imshow(image)
plt.title(f"Predicted: {predicted_label}")
plt.show()