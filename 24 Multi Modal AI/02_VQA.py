"""
Project 922. Visual Question Answering (VQA)

Visual Question Answering (VQA) systems answer questions related to the content of an image. It combines both image understanding and natural language processing to generate accurate responses. In this project, we simulate a simple image-based question answering system using a pre-trained model.

What This Does:
ViLT (Vision-and-Language Transformer) combines both vision and language inputs, enabling it to understand images and answer questions.

The model processes the question and image together to generate an answer.
"""

from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image
 
# Load pre-trained VQA model and processor
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
 
# Simulate a question and an image
image = Image.open("example_image.jpg")  # Replace with a valid image path
question = "What is in the image?"
 
# Preprocess the image and question
inputs = processor(text=question, images=image, return_tensors="pt", padding=True)
 
# Perform forward pass to get model's prediction
outputs = model(**inputs)
 
# Extract the answer from the model's output
answer = outputs.logits.argmax(-1)  # Get the index of the most likely answer
answer_str = processor.decode(answer)
 
print(f"Question: {question}")
print(f"Answer: {answer_str}")