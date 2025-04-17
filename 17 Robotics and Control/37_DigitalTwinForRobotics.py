"""
Project 677: Digital Twin for Robotics
Description:
A digital twin is a virtual representation of a physical system, in this case, a robot, that mirrors its real-world counterpart in real time. Digital twins allow for simulations, testing, and optimization of robot behavior without physical hardware. In this project, we will implement a digital twin for a simple robot that simulates its behavior and environment. The digital twin will simulate the robot's position, sensor data, and movement, which can be used for testing and optimizing the robot's control algorithms in a virtual environment.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Digital Twin class
class DigitalTwinRobot:
    def __init__(self, initial_position=(0, 0), target_position=(5, 5), map_size=(10, 10)):
        self.position = np.array(initial_position)  # Robot's initial position
        self.target_position = np.array(target_position)  # Goal position
        self.velocity = np.array([0.1, 0.1])  # Robot's initial velocity
        self.map_size = map_size  # Size of the map (grid)
        self.sensor_noise = 0.1  # Simulated sensor noise for measurements
        self.history = []  # Store the history of positions for visualization
 
    def move(self):
        """
        Simulate robot movement based on its velocity.
        """
        self.position += self.velocity
        # Ensure the robot stays within the bounds of the map
        self.position = np.clip(self.position, [0, 0], np.array(self.map_size) - 1)
 
    def get_sensor_data(self):
        """
        Simulate sensor data (distance from robot to target with added noise).
        :return: Simulated sensor data with noise (distance to target)
        """
        distance_to_target = np.linalg.norm(self.target_position - self.position)
        noisy_measurement = distance_to_target + np.random.randn() * self.sensor_noise
        return noisy_measurement
 
    def update(self):
        """
        Update the robot's state (position and sensor data).
        """
        self.move()  # Move the robot
        sensor_data = self.get_sensor_data()  # Get sensor data (distance to target)
        self.history.append(self.position.copy())  # Record the position
        return sensor_data
 
    def plot(self):
        """
        Visualize the robot's trajectory and the digital twin behavior.
        """
        history_array = np.array(self.history)
        plt.figure(figsize=(8, 8))
        plt.plot(history_array[:, 0], history_array[:, 1], label="Robot Path", color='blue')
        plt.scatter(self.target_position[0], self.target_position[1], color='red', s=100, label="Target Position")
        plt.scatter(self.position[0], self.position[1], color='green', s=100, label="Current Position")
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        plt.title("Digital Twin Robot Simulation")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
# 2. Initialize the digital twin robot and simulate its movement
digital_twin = DigitalTwinRobot(initial_position=(0, 0), target_position=(8, 8), map_size=(10, 10))
 
# 3. Simulate the robot's movement and sensor updates over 50 steps
for step in range(50):
    sensor_data = digital_twin.update()  # Update the robot's position and get sensor data
    if step % 10 == 0:  # Plot every 10 steps
        digital_twin.plot()  # Plot the current position and trajectory