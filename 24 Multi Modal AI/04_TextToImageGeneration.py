"""
Project 924. Text-to-Image Generation

Text-to-image generation creates images from textual descriptions, enabling creative content generation and design automation. In this project, we simulate a text-to-image system using DALL·E or Stable Diffusion, which takes a text prompt and generates a corresponding image.

What This Does:
Stable Diffusion is a powerful generative model that creates high-quality images from textual prompts.

It can generate scenes, characters, objects, or even abstract concepts based on text input.
"""

from diffusers import StableDiffusionPipeline
import torch
 
# Load pre-trained text-to-image generation model (Stable Diffusion)
model_id = "CompVis/stable-diffusion-v-1-4-original"  # You can change this model ID for variations
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU for faster generation
 
# Text prompt for image generation
prompt = "A futuristic city with flying cars and neon lights."
 
# Generate the image
image = pipe(prompt).images[0]
 
# Show the generated image
image.show()