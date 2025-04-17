"""
Project 659: Learning from Demonstration
Description:
Learning from demonstration (LfD), also known as imitation learning, allows robots to learn tasks by observing demonstrations performed by humans or other robots. The robot learns the control policies from the demonstrations without requiring explicit programming. In this project, we will implement a simple learning from demonstration system where the robot learns a task by mimicking human-provided trajectory data. We will use a supervised learning approach (e.g., regression or classification) to model the mapping from states to actions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
# 1. Simulate a simple robot task (e.g., moving to a target position)
# Generate human demonstration data (states and actions)
# States: (position)
# Actions: (desired velocity)
 
def generate_demonstration_data():
    # Generate demonstration data (position and corresponding velocity)
    positions = np.linspace(0, 10, 100)  # Simulate 100 positions from 0 to 10 meters
    velocities = np.sin(positions)  # Simple trajectory: velocity is a sine function of position (just for illustration)
    return positions, velocities
 
# 2. Define the robot's learning algorithm (supervised regression)
class LearningFromDemonstration:
    def __init__(self):
        self.model = LinearRegression()  # Using linear regression for simplicity (can be any regression model)
 
    def train(self, states, actions):
        """
        Train the model using human-provided demonstration data.
        :param states: Positions (states) from the demonstration
        :param actions: Velocities (actions) from the demonstration
        """
        states = states.reshape(-1, 1)  # Reshape for regression (1 feature)
        self.model.fit(states, actions)
 
    def predict(self, state):
        """
        Predict the action (velocity) for a given state (position).
        :param state: The state (position) for which to predict the action (velocity)
        :return: Predicted velocity (action)
        """
        return self.model.predict(np.array([[state]]))
 
# 3. Generate human demonstration data
positions, velocities = generate_demonstration_data()
 
# 4. Train the robot's model using the demonstration data
robot = LearningFromDemonstration()
robot.train(positions, velocities)
 
# 5. Simulate the robot's behavior using the learned model
predicted_velocities = [robot.predict(pos)[0] for pos in positions]
 
# 6. Plot the results: Demonstration data vs learned model
plt.figure(figsize=(10, 6))
 
# Plot demonstration data (actual velocity vs position)
plt.plot(positions, velocities, label="Demonstration Data (Actual)", color='blue')
 
# Plot predicted velocities using the learned model
plt.plot(positions, predicted_velocities, label="Learned Model (Predicted)", color='red', linestyle='--')
 
plt.title("Learning from Demonstration - Position vs Velocity")
plt.xlabel("Position (m)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid(True)
plt.show()