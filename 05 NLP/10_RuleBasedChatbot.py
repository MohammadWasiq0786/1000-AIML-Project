"""
Project 170. Chatbot using rule-based methods
Description:
A rule-based chatbot uses predefined patterns and responses to simulate a conversation. This project builds a simple chatbot using if-else logic or regular expressions to match user inputs and reply with fixed responses—ideal for FAQs, customer support, and tutorial use cases.

About:
🧠 What This Project Demonstrates:
Implements a basic rule-based chatbot with regex pattern matching

Matches user input to keywords or intents

Provides static but relevant responses
"""

import re
 
# Define chatbot rules and responses
def chatbot_response(user_input):
    user_input = user_input.lower().strip()
 
    # Greetings
    if re.search(r"\b(hi|hello|hey|greetings)\b", user_input):
        return "Hello! How can I help you today?"
 
    # Farewell
    elif re.search(r"\b(bye|goodbye|see you)\b", user_input):
        return "Goodbye! Have a great day!"
 
    # Asking for help
    elif "help" in user_input:
        return "Sure, I can assist you. Please describe your issue."
 
    # Weather inquiry
    elif re.search(r"weather", user_input):
        return "I'm not connected to a weather API yet, but I bet it's a beautiful day!"
 
    # Thanks
    elif re.search(r"\b(thanks|thank you)\b", user_input):
        return "You're welcome!"
 
    # Default fallback
    else:
        return "I'm not sure how to respond to that. Could you rephrase?"
 
# Run the chatbot in a loop
print("🤖 RuleBot: Hi! I’m RuleBot. Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("🤖 RuleBot: Goodbye! 👋")
        break
    response = chatbot_response(user_input)
    print("🤖 RuleBot:", response)