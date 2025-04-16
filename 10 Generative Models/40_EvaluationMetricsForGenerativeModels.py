"""
Project 400. Evaluation metrics for generative models
Description:
Evaluating the performance of generative models (such as GANs or VAEs) is crucial to understand how well the model generates realistic and diverse outputs. Various metrics are used to quantify the quality of the generated data, including Inception Score (IS), Fréchet Inception Distance (FID), Precision and Recall, and Maximum Mean Discrepancy (MMD).

In this project, we will implement and explore some of these evaluation metrics for generative models using pre-trained models like InceptionV3 for calculating FID and IS.

About:
✅ What It Does:
Inception Score (IS): Measures the clarity and diversity of the generated images by evaluating how well the images can be classified by a pre-trained InceptionV3 model.

Fréchet Inception Distance (FID): Measures the distance between the feature distributions of real and generated images. A lower FID indicates better performance, with the goal being to make generated images as close as possible to real images.

The generated images and real images are passed through the InceptionV3 model to extract features, and then the metrics are calculated based on these features.

Key features:
Inception Score quantifies how "realistic" and diverse the generated images are by checking their classification probabilities.

Fréchet Inception Distance evaluates how similar the generated image distribution is to that of the real images.

These metrics are essential for evaluating the performance of generative models like GANs and VAEs.
"""

# pip install torch torchvision numpy scipy

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import scipy.linalg
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
 
# 1. Load pre-trained InceptionV3 model for feature extraction
inception_model = models.inception_v3(pretrained=True, transform_input=False).eval()
 
# 2. Inception Score Calculation
def inception_score(images, model=inception_model, splits=10):
    # Convert images to appropriate format and normalize
    batch_size = len(images)
    images = images.clone().detach()
    images = F.interpolate(images, size=(299, 299))  # Resize to 299x299 (input size for InceptionV3)
    images = images * 2 - 1  # Normalize the images to [-1, 1]
    
    # Calculate inception features (using InceptionV3)
    with torch.no_grad():
        outputs = model(images)
        p_yx = F.softmax(outputs, dim=1).cpu().numpy()
        p_y = np.mean(p_yx, axis=0)
        kl_divergence = np.sum(p_yx * (np.log(p_yx) - np.log(p_y)), axis=1)
        return np.exp(np.mean(kl_divergence))
 
# 3. Fréchet Inception Distance Calculation
def calculate_fid(real_images, generated_images, model=inception_model):
    # Convert images to appropriate format
    real_images = F.interpolate(real_images, size=(299, 299))  # Resize to 299x299
    generated_images = F.interpolate(generated_images, size=(299, 299))  # Resize to 299x299
    
    # Normalize images
    real_images = (real_images * 2 - 1)
    generated_images = (generated_images * 2 - 1)
    
    # Extract features using InceptionV3
    with torch.no_grad():
        real_features = model(real_images).cpu().numpy()
        generated_features = model(generated_images).cpu().numpy()
 
    # Compute mean and covariance of the features for real and generated images
    real_mu, real_sigma = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    gen_mu, gen_sigma = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    
    # Compute Fréchet Distance (FID)
    diff = real_mu - gen_mu
    covmean, _ = scipy.linalg.sqrtm(real_sigma.dot(gen_sigma), disp=False)
    fid = np.sum(diff**2) + np.trace(real_sigma + gen_sigma - 2 * covmean)
    return fid
 
# 4. Example usage: Generating some random images (for demonstration purposes)
# Note: You would use generated images here (e.g., from a GAN) instead of random data
generated_images = torch.randn(64, 3, 64, 64)  # Random images (batch of 64)
real_images = torch.randn(64, 3, 64, 64)  # Real images (batch of 64)
 
# 5. Calculate Inception Score and FID
is_score = inception_score(generated_images)
fid_score = calculate_fid(real_images, generated_images)
 
print(f"Inception Score: {is_score:.4f}")
print(f"Fréchet Inception Distance (FID): {fid_score:.4f}")