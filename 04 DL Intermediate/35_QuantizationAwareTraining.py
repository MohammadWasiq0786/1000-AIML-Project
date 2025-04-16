"""
Project 155. Quantization-aware training
Description:
Quantization-Aware Training (QAT) simulates the effects of quantization during training, allowing models to learn to be robust to reduced precision (e.g., int8). Unlike post-training quantization, QAT yields higher accuracy and is better suited for edge devices, mobile, and IoT applications where low latency and small model size matter.

About:
🧠 What This Project Demonstrates:
Performs Quantization-Aware Training (QAT) with minimal code changes

Uses QuantStub/DeQuantStub for simulation of low-precision ops

Converts the model to a real int8 quantized model after training
"""

# Install if not already: pip install torch torchvision
 
import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 
# Prepare MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
 
# Define quantizable model
class QuantizedMLP(nn.Module):
    def __init__(self):
        super(QuantizedMLP, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        self.dequant = torch.quantization.DeQuantStub()
 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return self.dequant(x)
 
# Instantiate and prepare for QAT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantizedMLP().to(device)
 
# Fuse layers (important for quantization)
model.fuse_model = torch.quantization.fuse_modules(
    model,
    [['fc1', 'relu1'], ['fc2', 'relu2']]
) if hasattr(model, 'fuse_model') else None
 
# Prepare QAT config
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
 
# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
def train(model, loader):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
# Train with QAT
for epoch in range(3):
    train(model, train_loader)
    print(f"🧠 Epoch {epoch+1} complete (QAT)")
 
# Convert to quantized model
model.to('cpu')
quantized_model = torch.quantization.convert(model.eval(), inplace=False)
 
# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(loader.dataset)
 
acc = evaluate(quantized_model, test_loader)
print(f"📦 Final Quantized Model Accuracy: {acc:.2%}")