"""
Project 232. Image enhancement techniques
Description:
Image enhancement improves the visual quality of an image by making features more distinct or improving visibility. Common techniques include histogram equalization, contrast stretching, sharpening, and denoising. This project demonstrates key enhancement methods using OpenCV to transform low-quality images into clearer, more usable ones.

About:
What It Does:
This project improves images using four classic enhancement techniques. Whether you're working with dim lighting, blurry photos, or low-contrast scenes, these tools are essential in photography, medical imaging, satellite analysis, and digital forensics. You can further integrate AI-powered methods like RetinexNet or zero-DCE for deep learning-based enhancement.
"""
# Install required packages:
# pip install opencv-python matplotlib
 
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
# Load the input image (preferably a low-contrast or dim image)
image_path = "low_quality_image.jpg"  # Replace with your image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
# ---------- 1. Histogram Equalization (grayscale) ----------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist_eq = cv2.equalizeHist(gray)
 
# ---------- 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) ----------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
 
# ---------- 3. Image Sharpening ----------
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(image_rgb, -1, kernel)
 
# ---------- 4. Brightness and Contrast Adjustment ----------
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 30    # Brightness control (0-100)
adjusted = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
 
# ---------- Display Results ----------
plt.figure(figsize=(14, 8))
 
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(2, 3, 2)
plt.imshow(hist_eq, cmap='gray')
plt.title("Histogram Equalization")
plt.axis('off')
 
plt.subplot(2, 3, 3)
plt.imshow(clahe_img, cmap='gray')
plt.title("CLAHE")
plt.axis('off')
 
plt.subplot(2, 3, 4)
plt.imshow(sharpened)
plt.title("Sharpened Image")
plt.axis('off')
 
plt.subplot(2, 3, 5)
plt.imshow(adjusted)
plt.title("Brightness & Contrast Adjusted")
plt.axis('off')
 
plt.tight_layout()
plt.show()