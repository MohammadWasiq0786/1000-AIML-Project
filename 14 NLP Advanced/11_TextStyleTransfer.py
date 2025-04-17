"""
Project 531: Text Style Transfer
Description:
Text style transfer is a task in NLP where the goal is to transform the style of a given text without altering its content. Examples include converting formal text into casual language or transforming the sentiment of a piece of text. In this project, we will use transformer models to perform style transfer on text, such as converting a formal sentence into a more casual tone.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for text style transfer
style_transfer_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
 
# 2. Provide a formal sentence to be converted into a casual tone
formal_text = "It is with great pleasure that I write to inform you of our upcoming meeting."
 
# 3. Use the pipeline to generate a casual version of the text
casual_text = style_transfer_pipeline(formal_text)
 
# 4. Display the transformed text
print(f"Casual Version: {casual_text[0]['generated_text']}")