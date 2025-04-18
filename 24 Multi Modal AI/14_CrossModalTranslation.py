"""
Project 934. Cross-modal Translation

Cross-modal translation involves translating information between different modalities—such as translating image content into text or text into audio. For instance, converting a visual scene into a descriptive paragraph or translating a text description into a corresponding image.

In this project, we simulate text-to-image translation, where we generate an image from a text description and then describe the generated image using image captioning.

Step 1: Text-to-Image Translation
We’ll use Stable Diffusion to generate an image from the provided text.

Step 2: Image-to-Text Translation
We will use BLIP (Bootstrapping Language-Image Pre-training) to caption the generated image.

What This Does:
Text-to-Image Translation: Generates an image from the text description using Stable Diffusion.

Image-to-Text Translation: Uses BLIP to generate a caption for the generated image.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
 
# Load pre-trained Stable Diffusion model for text-to-image generation
stable_diff_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original", torch_dtype=torch.float16)
stable_diff_pipe.to("cuda")  # Move model to GPU for faster generation
 
# Load pre-trained BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
 
# Step 1: Generate an image from the text description using Stable Diffusion
text_prompt = "A beautiful sunset over the ocean with a sailboat on the horizon."
generated_image = stable_diff_pipe(text_prompt).images[0]
 
# Step 2: Caption the generated image using BLIP
inputs = blip_processor(images=generated_image, return_tensors="pt")
out = blip_model.generate(**inputs)
caption = blip_processor.decode(out[0], skip_special_tokens=True)
 
# Show the generated image and its caption
generated_image.show()
print(f"Generated Image Caption: {caption}")