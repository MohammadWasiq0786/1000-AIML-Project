"""
Project 469. Virtual health assistant
Description:
A Virtual Health Assistant acts as a conversational agent that helps users track health, manage medications, answer health questions, and even schedule appointments. In this project, we’ll implement a basic multi-intent chatbot that can answer medical questions, log symptoms, and remind about medications using intent classification and a simple response engine.

About:
✅ What It Does:
Classifies user input into basic health intents (symptoms, reminders, questions).

Responds appropriately to each intent.

Easily expandable to:

Add custom actions (e.g. API calls to calendar, drug info)

Use LLMs like BioGPT or MedPalm for deep Q&A

Deploy via web or mobile for patient interaction

For more advanced assistants:

Use Rasa, Dialogflow, or LangChain
For more advanced assistants:
Integrate EHR APIs, medication reminders, and LLMs
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
 
# 1. Example training data for intents
data = [
    ("I have a headache and nausea", "symptom_log"),
    ("Remind me to take my medicine at 8", "med_reminder"),
    ("What are the symptoms of diabetes?", "medical_question"),
    ("Can you tell me about flu symptoms?", "medical_question"),
    ("I feel tired and have a sore throat", "symptom_log"),
    ("Set a reminder for my blood pressure pills", "med_reminder"),
]
 
X_text, y_labels = zip(*data)
 
# 2. Train a simple intent classifier
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_text)
clf = MultinomialNB()
clf.fit(X_vectorized, y_labels)
 
# 3. Response engine based on intent
def respond(intent, message):
    if intent == "symptom_log":
        return f"Thanks. I’ve logged your symptoms: “{message}”. Would you like to schedule a checkup?"
    elif intent == "med_reminder":
        return "Your medication reminder has been set!"
    elif intent == "medical_question":
        return "Let me look that up... Here’s a summary of what I found: [placeholder response]"
    else:
        return "I'm not sure how to help with that yet, but I'm learning!"
 
# 4. Chat loop
print("Welcome to Vita! Your Virtual Health Assistant.\n(Type 'exit' to stop)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
 
    vec = vectorizer.transform([user_input])
    predicted_intent = clf.predict(vec)[0]
    response = respond(predicted_intent, user_input)
    print(f"Vita: {response}\n")