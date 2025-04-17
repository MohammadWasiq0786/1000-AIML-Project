"""
Project 549: Empathetic Dialogue System
Description:
An empathetic dialogue system aims to respond to user inputs with emotional understanding, providing responses that reflect empathy, support, and consideration of the user's emotional state. In this project, we will build a simple empathetic chatbot using a pre-trained model and add functionality for identifying and responding to emotional cues in user inputs.
"""

from transformers import pipeline
 
# 1. Load pre-trained model for empathetic response generation
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
 
# 2. Define a function to detect empathy cues in text and generate an empathetic response
def generate_empathetic_response(user_input):
    # Simple check for emotional words (e.g., sadness, happiness)
    if "sad" in user_input or "upset" in user_input:
        response = "I'm really sorry you're feeling this way. Is there anything I can do to help?"
    elif "happy" in user_input or "good" in user_input:
        response = "That's great to hear! I'm so glad you're feeling good today!"
    else:
        # Use the chatbot to generate a more general response
        response = chatbot(user_input)[0]['generated_text']
    return response
 
# 3. Simulate a conversation with empathetic responses
user_input = "I am feeling really sad today."
bot_response = generate_empathetic_response(user_input)
print(f"Bot: {bot_response}")
 
user_input = "I just got some great news!"
bot_response = generate_empathetic_response(user_input)
print(f"Bot: {bot_response}")