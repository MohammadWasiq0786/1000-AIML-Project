"""
Project 550: Task-Oriented Dialogue System
Description:
A task-oriented dialogue system is designed to help users accomplish specific tasks, such as booking a flight, ordering food, or setting a reminder. These systems focus on extracting user intent and fulfilling specific goals. In this project, we will create a simple task-oriented dialogue system that interacts with the user to complete a predefined task, such as flight booking.
"""

class FlightBookingSystem:
    def __init__(self):
        self.state = {
            "intent": None,
            "from": None,
            "to": None,
            "date": None,
        }
 
    def update_state(self, intent, slots):
        self.state["intent"] = intent
        self.state.update(slots)
 
    def get_state(self):
        return self.state
 
# 1. Instantiate the flight booking system
booking_system = FlightBookingSystem()
 
# 2. Simulate user interaction (providing details to book a flight)
user_input_1 = "I want to book a flight from New York to Paris."
intent_1 = "book_flight"
slots_1 = {"from": "New York", "to": "Paris", "date": "tomorrow"}
 
# Update the system's state
booking_system.update_state(intent_1, slots_1)
 
# 3. Retrieve and display the current state of the booking system
current_state = booking_system.get_state()
print(f"Current Flight Booking State: {current_state}")
 
# 4. Simulate another user input (e.g., user wants to change the flight date)
user_input_2 = "Can I book it for next Monday instead?"
intent_2 = "change_date"
slots_2 = {"date": "next Monday"}
 
# Update the system's state again
booking_system.update_state(intent_2, slots_2)
 
# 5. Retrieve and display the updated state
updated_state = booking_system.get_state()
print(f"Updated Flight Booking State: {updated_state}")