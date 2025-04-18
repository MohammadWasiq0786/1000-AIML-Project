"""
Project 950. Multi-modal Fake News Detection

Multi-modal fake news detection systems use both textual and visual cues to determine whether news content is authentic or fake. These systems combine linguistic features (from the text) and visual features (from images or videos) to improve the accuracy of fake news classification.

In this project, we simulate multi-modal fake news detection by analyzing textual content (e.g., articles or headlines) and visual content (e.g., images associated with the news) to classify the news as fake or real.

Step 1: Textual Feature Extraction
We will use BERT (or another transformer-based model) for text classification to analyze whether the content is likely to be fake or real.

Step 2: Visual Feature Extraction
We will use CLIP to analyze whether the associated image matches the claims in the text (e.g., does the image support the narrative, or is it misleading?).

Step 3: Multi-modal Fake News Classification
We combine both text and image features to perform fake news classification.

What This Does:
Textual Analysis: Uses a pre-trained BERT model to classify the news text as fake or real based on linguistic features.

Visual Analysis: Uses CLIP to check the consistency between the text and the associated image. A low similarity score between the text and image may indicate image manipulation or misleading visuals, which could suggest fake news.

Multi-modal Classification: Combines the results from both the text classifier and image similarity to classify the news as fake or real.
"""

from transformers import pipeline, CLIPProcessor, CLIPModel
import torch
from PIL import Image
 
# Load pre-trained BERT model for text classification (fake news detection)
fake_news_classifier = pipeline("text-classification", model="bert-base-uncased")
 
# Load pre-trained CLIP model and processor for image analysis
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
 
# Step 1: Fake News Classification from Text
def classify_fake_news_from_text(text):
    result = fake_news_classifier(text)
    return result[0]['label']
 
# Step 2: Visual Mismatch Detection using CLIP
def analyze_image_and_text(text, image_path):
    # Load image and process with CLIP
    image = Image.open(image_path)
    inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    # Perform forward pass to get image-text embeddings
    outputs = clip_model(**inputs)
    image_embeddings = outputs.image_embeds
    text_embeddings = outputs.text_embeds
 
    # Calculate similarity between the text and image embeddings
    similarity_score = torch.cosine_similarity(text_embeddings, image_embeddings)
    return similarity_score.item()
 
# Example inputs
news_text = "The recent viral video shows a protest in downtown, claiming that the government is corrupt."
image_path = "fake_image.jpg"  # Replace with a valid image path
 
# Step 1: Classify fake news based on text
news_classification = classify_fake_news_from_text(news_text)
print(f"News Classification (Text-based): {news_classification}")
 
# Step 2: Analyze image-text consistency using CLIP
image_text_similarity = analyze_image_and_text(news_text, image_path)
print(f"Image-Text Similarity Score: {image_text_similarity:.2f}")
 
# Step 3: Fake News Decision based on both modalities
if news_classification == "LABEL_1" and image_text_similarity < 0.5:
    print("Conclusion: Fake news detected based on text and image inconsistency.")
else:
    print("Conclusion: News seems authentic.")