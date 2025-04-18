"""
Project 939. Multi-modal Reasoning System

A multi-modal reasoning system can make decisions or infer knowledge by combining information from multiple modalities such as text, images, audio, and video. In this project, we simulate a simple multi-modal reasoning system that combines textual descriptions and visual content (images) to perform reasoning tasks, such as answering questions or making inferences.

We’ll use CLIP (Contrastive Language-Image Pre-training) for image-text reasoning and transformers for generating textual reasoning.

Step 1: Image-Text Reasoning
We’ll use CLIP to generate embeddings for both the image and text and compare their similarity to answer a question related to the image.

Step 2: Simple Question Answering (QA)
We will simulate a reasoning task where the model answers a question about the image using the textual context and visual content.

What This Does:
Image-Text Reasoning: We use CLIP to measure the similarity between an image and different textual descriptions/questions, answering which question is most relevant to the image.

Multi-modal Reasoning: The system combines text and visual features to reason about the content of the image.
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Simulate a question and an image dataset
image = Image.open("example_image.jpg")  # Replace with a valid image path
questions = ["What is the object in the image?", "Is this object a vehicle?", "What color is the object?"]
 
# Preprocess the image and text inputs
def multi_modal_reasoning(image, questions):
    inputs = processor(text=questions, images=image, return_tensors="pt", padding=True)
 
    # Perform forward pass to get model's prediction
    outputs = model(**inputs)
 
    # Calculate similarity between the image and each text question
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
 
    # Retrieve the most relevant question and answer
    best_match_idx = torch.argmax(probs)
    return questions[best_match_idx], probs[0][best_match_idx].item()
 
# Simulate reasoning task
best_question, match_score = multi_modal_reasoning(image, questions)
 
# Output the result
print(f"Image-Text Reasoning Result: {best_question}")
print(f"Match Score: {match_score:.2f}")