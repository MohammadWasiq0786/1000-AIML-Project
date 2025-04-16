"""
Project 182. Dialogue state tracking
Description:
Dialogue State Tracking (DST) maintains the state of a conversation by keeping track of the user's intents, slots (extracted info), and system actions across multiple turns. This enables a conversational AI to handle complex, multi-turn dialogues and make decisions based on the current dialogue context.

About:
🧠 What This Project Demonstrates:
Tracks the intent and slot values throughout the conversation

Maintains a persistent state dictionary for dialogue context

Uses SpaCy NER and regex to update the state turn-by-turn
"""

# Simulated dialogue between user and bot for a restaurant booking
 
# Initialize an empty dialogue state
dialogue_state = {
    "intent": None,
    "slots": {
        "location": None,
        "cuisine": None,
        "party_size": None,
        "reservation_time": None
    }
}
 
# Simulated user inputs (conversation turns)
user_turns = [
    "Hi, I'd like to book a table.",
    "It should be in New York.",
    "We want Chinese food.",
    "There are four of us.",
    "Make it for 7 pm tonight."
]
 
# Simple keyword + pattern-based slot extraction
import re
import spacy
nlp = spacy.load("en_core_web_sm")
 
def extract_slots(text):
    doc = nlp(text)
    slots = {}
 
    # Extract location
    for ent in doc.ents:
        if ent.label_ == "GPE":
            slots["location"] = ent.text
 
    # Cuisine keywords
    if "chinese" in text.lower():
        slots["cuisine"] = "Chinese"
    elif "italian" in text.lower():
        slots["cuisine"] = "Italian"
 
    # Party size
    match = re.search(r'\b(\d+)\b', text)
    if match:
        slots["party_size"] = match.group(1)
 
    # Time
    for ent in doc.ents:
        if ent.label_ == "TIME" or ent.label_ == "DATE":
            slots["reservation_time"] = ent.text
 
    return slots
 
# Simulate dialogue and update state
for turn in user_turns:
    extracted = extract_slots(turn)
    for key, value in extracted.items():
        if value:
            dialogue_state["slots"][key] = value
 
# Final intent (hardcoded for simplicity here)
dialogue_state["intent"] = "book_table"
 
# Display the final state
print("🧠 Final Dialogue State:\n")
print(f"Intent: {dialogue_state['intent']}")
print("Slots:")
for slot, value in dialogue_state["slots"].items():
    print(f" - {slot}: {value if value else 'Not filled'}")