"""
Project 937. Multi-modal Summarization

Multi-modal summarization systems generate concise summaries that capture key information from multiple modalities, such as text, images, and audio. In this project, we simulate multi-modal summarization by combining text summarization with image captioning, which helps summarize both the content (from text) and visual aspects (from images).

Step 1: Text Summarization
We'll use a pre-trained transformer model like BART or T5 for summarizing the text.

Step 2: Image Captioning
We'll use the BLIP model for generating captions for images, summarizing what’s visually important.

What This Does:
Text Summarization: Uses BART (or T5) to generate a concise summary of a given text.

Image Captioning: Uses BLIP to generate a description of the visual content in an image.

Multi-modal Summary: Combines both the text summary and image caption into a single coherent summary.
"""

from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
 
# Step 1: Text Summarization using T5 or BART
text_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
 
def summarize_text(text):
    summary = text_summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']
 
# Example text input
text_input = """
The Eiffel Tower is one of the most famous landmarks in Paris, France. It was designed by engineer Gustave Eiffel and completed in 1889. 
The tower stands at 324 meters and was initially met with skepticism, but it became an iconic symbol of France's ingenuity and elegance. 
Today, it attracts millions of tourists every year who come to see its breathtaking views of the city.
"""
text_summary = summarize_text(text_input)
print(f"Text Summary: {text_summary}")
 
# Step 2: Image Captioning using BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
 
def caption_image(image_path):
    image = Image.open(image_path)
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption
 
# Example image path (replace with a valid image path)
image_path = "example_image.jpg"  # Replace with a valid image
image_caption = caption_image(image_path)
print(f"Image Caption: {image_caption}")
 
# Step 3: Combine Text and Image Summaries
combined_summary = f"Text Summary: {text_summary}\nImage Caption: {image_caption}"
print(f"\nCombined Multi-modal Summary: {combined_summary}")