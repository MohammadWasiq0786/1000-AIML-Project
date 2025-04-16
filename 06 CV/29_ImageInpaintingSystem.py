"""
Project 229. Image inpainting system
Description:
Image inpainting is the process of filling in missing or corrupted parts of an image in a visually plausible way. It’s useful in photo restoration, object removal, and editing applications. In this project, we’ll use a pre-trained inpainting model from Hugging Face’s diffusers library (like LaMa or SD-Inpainting) to automatically complete masked regions.

About:
What It Does:
This project can reconstruct damaged or edited regions in an image using advanced generative AI. Applications include photo restoration, unwanted object removal, hole filling, and even creative editing. You can enhance it with models like LaMa, Paint-by-Example, or use OpenCV-based traditional inpainting for lightweight tasks.
"""
# Install required packages:
# pip install diffusers transformers torch torchvision accelerate pillow matplotlib
 
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt
 
# Load the pre-trained inpainting pipeline (Stable Diffusion Inpainting model)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
).to("cuda")
 
# Load original image and the mask image
# The mask should be a black-and-white image: white = area to inpaint, black = keep
image = Image.open("damaged_image.png").convert("RGB")        # Replace with your image
mask = Image.open("inpaint_mask.png").convert("RGB")          # Replace with your mask
 
# Provide a text prompt for guidance (optional, makes it generative)
prompt = "Restore the missing part naturally"
 
# Run the inpainting model
output = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
 
# Display original, mask, and inpainted result
plt.figure(figsize=(15, 5))
 
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
 
plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("Inpainting Mask")
plt.axis('off')
 
plt.subplot(1, 3, 3)
plt.imshow(output)
plt.title("Inpainted Output")
plt.axis('off')
 
plt.tight_layout()
plt.show()