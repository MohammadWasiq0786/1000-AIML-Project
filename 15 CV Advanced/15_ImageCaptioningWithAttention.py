"""
Project 575: Image Captioning with Attention
Description:
Image captioning with attention involves generating a natural language description of an image, where the attention mechanism helps the model focus on specific regions of the image while generating the caption. In this project, we will use a pre-trained model like Show, Attend and Tell to generate captions with an attention mechanism.
"""

import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
 
# 1. Load pre-trained Vision-to-Text model (e.g., Show, Attend and Tell)
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
 
# 2. Load an image for captioning
image = Image.open("path_to_image.jpg")  # Replace with an actual image path
 
# 3. Preprocess the image and prepare inputs
inputs = processor(images=image, return_tensors="pt")
 
# 4. Generate the caption using the model
outputs = model.generate(input_ids=None, decoder_start_token_id=model.config.pad_token_id, 
                         **inputs, max_length=50, num_beams=5, early_stopping=True)
 
# 5. Decode and display the generated caption
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Caption: {caption}")