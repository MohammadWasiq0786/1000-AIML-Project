"""
Project 556: Bias Detection in Text
Description:
Bias detection involves identifying biased, prejudiced, or unfair representations in text, such as gender bias, racial bias, or ideological bias. In this project, we will use a pre-trained transformer model to detect biases in text and classify it based on the type of bias (if any).
"""

from transformers import pipeline
 
# 1. Load pre-trained model for bias detection
classifier = pipeline("text-classification", model="unitary/bias-bert")
 
# 2. Provide a text input to check for bias
text = "Women are not as good at math as men."
 
# 3. Classify the input text for bias
result = classifier(text)
 
# 4. Display the result
if result[0]['label'] == 'LABEL_1':  # LABEL_1 typically corresponds to biased content
    print(f"Bias Detected: {text}")
else:
    print(f"No Bias Detected: {text}")