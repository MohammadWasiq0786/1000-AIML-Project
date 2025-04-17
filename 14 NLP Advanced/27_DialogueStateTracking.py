"""
Project 547: Dialogue State Tracking
Description:
Dialogue state tracking is a core component of task-oriented dialogue systems, where the system keeps track of the context and the state of the conversation. This project involves implementing a system that can monitor and maintain the dialogue state, which includes user intents, slot values, and previous interactions. We will simulate a state tracking model that can be used to track user goals in a conversation.
"""

class DialogueStateTracker:
    def __init__(self):
        self.state = {
            "intent": None,
            "slots": {}
        }
 
    def update_state(self, intent, slots):
        self.state["intent"] = intent
        self.state["slots"] = slots
 
    def get_state(self):
        return self.state
 
# 1. Instantiate the dialogue state tracker
tracker = DialogueStateTracker()
 
# 2. Simulate an example conversation
user_input_1 = "Book a flight from New York to Paris."
intent_1 = "book_flight"
slots_1 = {"from": "New York", "to": "Paris", "date": "tomorrow"}
 
# Update the dialogue state with user input
tracker.update_state(intent_1, slots_1)
 
# 3. Retrieve and display the current state
current_state = tracker.get_state()
print(f"Current Dialogue State: {current_state}")
 
# 4. Simulate another user input
user_input_2 = "I want to depart in the evening."
intent_2 = "set_departure_time"
slots_2 = {"departure_time": "evening"}
 
# Update the dialogue state again
tracker.update_state(intent_2, slots_2)
 
# 5. Retrieve and display the updated state
updated_state = tracker.get_state()
print(f"Updated Dialogue State: {updated_state}")