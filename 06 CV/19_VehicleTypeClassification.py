"""
Project 219. Vehicle type classification
Description:
Vehicle Type Classification is the task of identifying the category of a vehicle—such as car, truck, bus, motorcycle, or van—from an image. It’s widely used in traffic monitoring, surveillance systems, smart toll booths, and parking analytics. In this project, we'll use a pre-trained CNN to classify vehicle types from static images.

About:
What It Does:
This project classifies vehicles into predefined categories, helping automate fleet management, smart traffic enforcement, license-based tolling, and vehicle analytics. The model can be expanded with a larger dataset and fine-tuned using pre-trained CNNs like ResNet or EfficientNet for higher accuracy.
"""
# Install required packages:
# pip install torch torchvision matplotlib pillow
 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
 
# Define a simple CNN architecture (or load a pre-trained one)
# This is a basic example for classification into 5 vehicle types
class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(VehicleClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
 
# Class labels for vehicle types
vehicle_classes = ["Car", "Truck", "Bus", "Motorcycle", "Van"]
 
# Initialize the model and load trained weights
model = VehicleClassifier()
model.load_state_dict(torch.load("vehicle_classifier.pth", map_location='cpu'))  # Replace with your model path
model.eval()
 
# Load and preprocess the vehicle image
image_path = "vehicle_image.jpg"  # Replace with your image
image = Image.open(image_path).convert("RGB")
 
transform = transforms.Compose([
    transforms.Resize((128, 128)),       # Resize to match input size
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])
 
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
 
# Predict vehicle type
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = outputs.argmax(1).item()
 
# Show result
plt.imshow(image)
plt.title(f"Predicted: {vehicle_classes[predicted_class]}")
plt.axis('off')
plt.show()
 
# Print prediction
print("\n🚗 Detected Vehicle Type:")
print(vehicle_classes[predicted_class])