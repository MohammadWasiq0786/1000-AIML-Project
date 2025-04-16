"""
Project 135. CLIP for image-text matching
Description:
CLIP (Contrastive Language–Image Pretraining) by OpenAI learns to understand images and natural language by jointly training on image–text pairs. In this project, we use a pre-trained CLIP model to match an image to its most relevant textual descriptions (or vice versa), enabling applications like zero-shot classification, caption ranking, and visual search.

About:
🧠 What This Project Demonstrates:
Uses CLIP to match images and text in a shared embedding space

Computes cosine similarity to rank text prompts against the image

Demonstrates zero-shot learning (no fine-tuning needed!)
"""

# Install if not already: pip install torch torchvision transformers openai-clip matplotlib
 
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
 
# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
# Load and preprocess image
image_path = "cat.jpg"  # Replace with your own image
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
 
# Candidate text descriptions
text_descriptions = [
    "A photo of a cat",
    "A photo of a dog",
    "A bowl of food",
    "A sunny beach",
    "A person riding a bike"
]
 
# Tokenize text inputs
text_tokens = clip.tokenize(text_descriptions).to(device)
 
# Encode image and text with CLIP
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
 
    # Compute cosine similarity
    logits_per_image, _ = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
# Display results
plt.imshow(Image.open(image_path))
plt.axis("off")
plt.title("📷 Input Image")
plt.show()
 
# Print matched results
print("🧠 Top Text Matches for Image:")
for i, (desc, prob) in enumerate(zip(text_descriptions, probs[0])):
    print(f"{i+1}. {desc} – {prob:.2%}")