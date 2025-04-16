"""
Project 123. StyleGAN for face generation
Description:
StyleGAN (Style-based Generative Adversarial Network) is a powerful generative model developed by NVIDIA for synthesizing high-quality, realistic human faces. In this project, we use a pre-trained StyleGAN2 model via the stylegan2_pytorch or rosinality implementation to generate fake faces from random latent vectors.

About:
🧠 What This Project Demonstrates:
Loads a StyleGAN2 model trained on human faces (FFHQ dataset)

Generates high-resolution, photorealistic face images

Works from a random latent vector, enabling infinite variations
"""

# Install if not already:
# pip install torch torchvision
# And clone the StyleGAN2-ADA PyTorch repo: https://github.com/NVlabs/stylegan2-ada-pytorch
 
import torch
import legacy
import dnnlib
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
 
# Load pre-trained StyleGAN2 model (FFHQ - Faces)
network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
 
print("Loading networks...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Load pre-trained generator
 
# Generate latent vector (z)
z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
 
# Generate face image
img = G(z, None, truncation_psi=0.7, noise_mode='const')[0]  # Output is (C, H, W)
img = (img.clamp(-1, 1) + 1) / 2  # Normalize to [0,1]
save_image(img, "generated_face.png")
 
# Display generated image
img_np = img.permute(1, 2, 0).cpu().detach().numpy()
plt.imshow(img_np)
plt.axis("off")
plt.title("🧑 AI-Generated Face (StyleGAN2)")
plt.show()