"""
Project 555: Hate Speech Detection
Description:
Hate speech detection is the task of identifying harmful or offensive content, typically in online communication, such as social media or forums. In this project, we will use a transformer model to detect hate speech in text and classify it as hate speech or non-hate speech.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for hate speech detection
classifier = pipeline("text-classification", model="unitary/toxic-bert")
 
# 2. Provide a text input to check for hate speech
text = "You are a worthless person and should not be here."
 
# 3. Classify the input text for hate speech
result = classifier(text)
 
# 4. Display the result
if result[0]['label'] == 'LABEL_1':  # LABEL_1 typically corresponds to toxic or hate speech
    print(f"Hate Speech Detected: {text}")
else:
    print(f"Non-Hate Speech: {text}")