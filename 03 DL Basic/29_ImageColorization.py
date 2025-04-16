"""
Project 109. Image colorization
Description:
Image colorization is the process of automatically adding color to grayscale images using AI. In this project, we build a deep learning model (a simplified CNN or U-Net) that learns to predict the color components of an image from its grayscale version. For quick prototyping, we’ll demonstrate using OpenCV and a pre-trained Caffe model for colorization.

About:
🎨 What This Project Demonstrates:
Takes grayscale input and predicts color components

Uses pre-trained deep learning model (Caffe-based) via OpenCV

Produces visually appealing colorized images
"""

# Install if not already: pip install opencv-python opencv-contrib-python
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
# Load pre-trained Caffe model and config
prototxt = cv2.samples.findFile("dnn/colorization_deploy_v2.prototxt")
model = cv2.samples.findFile("dnn/colorization_release_v2.caffemodel")
pts = cv2.samples.findFile("dnn/pts_in_hull.npy")
 
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts_data = np.load(pts)
 
# Add cluster centers to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts_data.astype(np.float32).reshape(2, 313, 1, 1)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
 
# Load a grayscale image and convert to LAB format
img_path = cv2.samples.findFile("messi5.jpg")  # Replace with your grayscale image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
 
# Resize and normalize the L channel
L = img_lab[:, :, 0]
L_resized = cv2.resize(L, (224, 224))
L_resized = L_resized - 50  # mean-centering
 
# Prepare blob and run model
net.setInput(cv2.dnn.blobFromImage(L_resized))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # (H, W, 2)
 
# Resize ab and concatenate with original L
ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
 
# Convert back to BGR
colorized_bgr = cv2.cvtColor(colorized.astype("float32"), cv2.COLOR_LAB2BGR)
colorized_bgr = np.clip(colorized_bgr, 0, 1)
 
# Display original and colorized image
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Grayscale")
plt.imshow(img, cmap="gray")
plt.axis("off")
 
plt.subplot(1, 2, 2)
plt.title("Colorized")
plt.imshow(colorized_bgr)
plt.axis("off")
plt.tight_layout()
plt.show()