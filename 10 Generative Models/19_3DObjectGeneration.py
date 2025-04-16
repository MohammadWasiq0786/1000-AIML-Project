"""
Project 379. 3D object generation
Description:
3D Object Generation is the process of creating three-dimensional objects from a latent space or other types of input. These models are commonly used in applications like 3D printing, augmented reality (AR), virtual reality (VR), and gaming. Generative models such as GANs, Variational Autoencoders (VAEs), and PointNet-based networks have been used to generate 3D objects.

In this project, we will implement a basic 3D object generation model using a point cloud approach, where the model learns to generate 3D point clouds representing objects.

About:
✅ What It Does:
PointCloudGenerator creates a 3D point cloud from a latent vector (random noise) using fully connected layers.

Chamfer Distance is used as a loss function to compare generated point clouds with real point clouds (here, random point clouds for simplicity).

The model learns to generate 3D objects by producing 3D point clouds that represent objects in 3D space.

Uses Open3D for handling and visualizing 3D data (point clouds).
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d  # Open3D for handling 3D data
import matplotlib.pyplot as plt
 
# 1. Define a simple 3D Object Generation model using a fully connected network
class PointCloudGenerator(nn.Module):
    def __init__(self, z_dim=100, num_points=1024):
        super(PointCloudGenerator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, num_points * 3)  # Output 3D point cloud (X, Y, Z)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
 
    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.tanh(x).view(-1, 1024, 3)  # Reshape to 3D points (1024 points with 3 coordinates)
 
# 2. Loss function and optimizer
def chamfer_distance(point_cloud1, point_cloud2):
    # Calculate Chamfer Distance (a commonly used distance metric for point clouds)
    distance_matrix = torch.cdist(point_cloud1, point_cloud2, p=2)  # Euclidean distance
    return torch.mean(torch.min(distance_matrix, dim=1)[0]) + torch.mean(torch.min(distance_matrix, dim=2)[0])
 
optimizer = optim.Adam(generator.parameters(), lr=0.0002)
 
# 3. Create random latent vectors for input
z_dim = 100  # Latent vector dimension
num_points = 1024  # Number of points in the point cloud
z = torch.randn(64, z_dim)  # Random noise input for batch size of 64
 
# 4. Training loop for 3D Object Generation model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = PointCloudGenerator(z_dim=z_dim, num_points=num_points).to(device)
 
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass through the generator
    generated_point_clouds = generator(z.to(device))
    
    # Calculate Chamfer distance loss between generated and real (target) point clouds
    real_point_clouds = torch.randn(64, num_points, 3).to(device)  # Random real point clouds (for simplicity)
    
    loss = chamfer_distance(generated_point_clouds, real_point_clouds)
    
    # Backpropagate the loss
    loss.backward()
    
    # Update the model
    optimizer.step()
 
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Generate and display 3D point clouds every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            # Visualize the generated 3D point cloud
            generated_point_cloud = generated_point_clouds[0].cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(generated_point_cloud[:, 0], generated_point_cloud[:, 1], generated_point_cloud[:, 2])
            plt.show()