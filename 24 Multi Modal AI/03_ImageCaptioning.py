"""
Project 923. Image Captioning Implementation

Image captioning generates textual descriptions for images, enabling applications like automatic image tagging, visual storytelling, and accessible content for the visually impaired. In this project, we simulate image captioning using a pre-trained image-to-text model.

What This Does:
BLIP (Bootstrapping Language-Image Pre-training) is a state-of-the-art model that generates captions for images by leveraging vision and language understanding.

It processes an image and generates a relevant textual description of its contents.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
 
# Load pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
 
# Simulate image for captioning
image = Image.open("example_image.jpg")  # Replace with a valid image path
 
# Preprocess the image and generate caption
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
 
# Decode the generated caption
caption = processor.decode(out[0], skip_special_tokens=True)
 
print(f"Generated Caption: {caption}")