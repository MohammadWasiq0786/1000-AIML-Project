"""
Project 230. Image colorization
Description:
Image colorization is the process of automatically adding realistic colors to grayscale images. It’s used in restoring black-and-white photos, enhancing historical archives, and artistic editing. In this project, we’ll use a pre-trained deep learning model from Hugging Face that colorizes grayscale images using a vision transformer.

About:
What It Does:
This project brings life to black-and-white photos by predicting realistic color schemes using deep learning. It’s ideal for photo restoration, historical media enhancement, or artistic filters. You can improve it with models like DeOldify, or use custom U-Net models trained on datasets like ImageNet or COCO for more tailored results.
"""
# Install required packages:
# pip install transformers diffusers torch torchvision pillow matplotlib
 
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
 
# Load a pre-trained colorization pipeline
colorizer = pipeline("image-to-image", model="JuliusKunze/colorization", device=0)  # Uses GPU if available
 
# Load a grayscale image (should be RGB or L mode)
gray_image = Image.open("black_white_photo.jpg").convert("L")  # Replace with your grayscale image
gray_image = gray_image.convert("RGB")  # Ensure 3 channels for compatibility
 
# Run the colorization pipeline
result = colorizer(gray_image)[0]['image']
 
# Display original and colorized images
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Grayscale")
plt.axis('off')
 
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Colorized Output")
plt.axis('off')
 
plt.tight_layout()
plt.show()