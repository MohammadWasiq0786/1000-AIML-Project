"""
Project 466. Stress level detection
Description:
Stress level detection leverages AI to recognize physical or emotional stress through text, voice, or physiological signals (e.g., heart rate, EDA, HRV). In this project, we simulate a text-based stress classifier using a pretrained sentiment/emotion model to predict stress level from user input.

✅ What It Does:
Accepts user’s free-text input (e.g., daily check-in, journal entry).

Uses a BERT-based emotion classifier to detect stress-indicating emotions.

Can be extended to:

Include voice tone, heart rate, or eye tracking

Provide real-time coping tips or referrals

Integrate into mental health chatbots or HR wellness tools

In production:

Use multi-modal signals (text + voice + biosignals)

Explore WESAD, DEAP, or SEED datasets for sensor-based stress detection
"""

from transformers import pipeline
 
# 1. Load emotion classification pipeline
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
 
# 2. Simulate interaction
print("Stress Check-In Tool: Type how you're feeling right now. Type 'exit' to quit.\n")
 
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
 
    # Predict emotional state
    result = classifier(user_input)[0]
    emotion = result["label"]
    confidence = result["score"]
 
    # Define stress-related emotions
    stress_labels = ["anger", "fear", "sadness", "disgust", "anxiety"]
 
    # Evaluate stress level
    if emotion in stress_labels:
        print(f"Detected Emotion: {emotion.upper()} (Stress likely) [Confidence: {confidence:.2f}]")
    else:
        print(f"Detected Emotion: {emotion.capitalize()} (No stress detected) [Confidence: {confidence:.2f}]")
    print("-" * 60)