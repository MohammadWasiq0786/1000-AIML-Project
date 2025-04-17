"""
Project 546: Conversational AI System
Description:
A conversational AI system allows machines to engage in dialogue with humans in a natural, interactive manner. This system can be built using pre-trained models like GPT, BERT, or specialized architectures like DialoGPT. In this project, we will build a simple conversational AI system using a pre-trained model that can generate responses to user inputs.
"""

from transformers import pipeline
 
# 1. Load pre-trained DialoGPT model for conversational AI
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
 
# 2. Define a function to simulate conversation with the chatbot
def chat_with_bot(user_input):
    # Provide the user's input to the chatbot
    response = chatbot(user_input)
    return response[0]['generated_text']
 
# 3. Example conversation with the chatbot
user_input = "Hello, how are you?"
bot_response = chat_with_bot(user_input)
print(f"Bot: {bot_response}")
 
# Continue the conversation
user_input = "What's the weather like today?"
bot_response = chat_with_bot(user_input)
print(f"Bot: {bot_response}")