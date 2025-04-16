"""
Project 228. Super-resolution for images
Description:
Super-resolution enhances the quality and detail of a low-resolution image by predicting a high-resolution version. It’s used in medical imaging, satellite imagery, facial enhancement, and video upscaling. In this project, we'll use a pre-trained Super-Resolution Convolutional Neural Network (SRCNN) model or ESPCN from PyTorch to upscale images.

About:
What It Does:
This project takes a blurry or pixelated image and enhances its quality with AI, recovering fine textures and sharper details. It’s a powerful tool in fields like surveillance, film restoration, and forensic imaging, and can be extended using advanced models like Real-ESRGAN or SwinIR for higher realism.
"""
# Install required packages:
# pip install torch torchvision pillow matplotlib
 
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18  # Just to access the hub
from torchvision.models import efficientnet_b0
from torchvision.models._api import WeightsEnum
from torchvision.models import video, resnet
from torchvision.models import segmentation
from torchvision.models import resnet34
from torchvision.models import super_resolution
from PIL import Image
import matplotlib.pyplot as plt
 
# Load the pre-trained super-resolution model from torchvision
# ESPCN = Efficient Sub-Pixel Convolutional Network
model = super_resolution.espcn(scale_factor=3, weights=super_resolution.ESPCN_X3Weights.DEFAULT)
model.eval()
 
# Define image transformation (convert to YCbCr and normalize Y channel)
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image.width // 3 * 3, image.height // 3 * 3))  # Ensure divisible by 3
    ycbcr = image.convert('YCbCr')
    y, cb, cr = ycbcr.split()
 
    transform = transforms.ToTensor()
    input_tensor = transform(y).unsqueeze(0)  # Add batch dimension
    return input_tensor, cb, cr, image
 
# Upscale using the ESPCN model
def upscale_image(image_path):
    input_tensor, cb, cr, original_image = preprocess_image(image_path)
 
    with torch.no_grad():
        output = model(input_tensor)
 
    output = output.squeeze().clamp(0, 1)  # Remove batch dimension and clamp pixel values
    out_y = transforms.ToPILImage()(output)
 
    # Resize Cb and Cr channels to match upscaled Y
    cb_up = cb.resize(out_y.size, Image.BICUBIC)
    cr_up = cr.resize(out_y.size, Image.BICUBIC)
 
    # Merge Y, Cb, Cr channels back to get final high-res RGB image
    final_image = Image.merge("YCbCr", [out_y, cb_up, cr_up]).convert("RGB")
    return original_image, final_image
 
# Run the super-resolution
low_res, high_res = upscale_image("low_res_sample.jpg")  # Replace with your image
 
# Display both images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(low_res)
plt.title("Original Low-Resolution")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(high_res)
plt.title("Enhanced Super-Resolution")
plt.axis('off')
plt.tight_layout()
plt.show()