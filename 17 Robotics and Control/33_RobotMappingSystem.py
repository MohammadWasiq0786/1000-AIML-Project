"""
Project 673: Robot Mapping System
Description:
A robot mapping system is essential for robots to understand their environment and create a map based on sensory data. The process of Simultaneous Localization and Mapping (SLAM) involves building a map while simultaneously determining the robot's position on that map. In this project, we will implement a basic 2D mapping system where a robot collects sensor data (e.g., distance measurements) and uses it to build a map of its surroundings. The robot will use a simple grid-based representation of the environment and move around while updating the map.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot Mapping System class
class RobotMappingSystem:
    def __init__(self, grid_size=(10, 10), robot_position=(5, 5), sensor_range=3.0):
        self.grid_size = grid_size  # Size of the map (grid size)
        self.robot_position = np.array(robot_position)  # Initial robot position
        self.sensor_range = sensor_range  # Range of the robot's sensor
        self.map = np.zeros(grid_size)  # Empty map (0 = free space, 1 = obstacle)
        self.history = []  # Store history of robot positions
 
    def get_sensor_data(self):
        """
        Simulate sensor data by detecting obstacles within the sensor range.
        In this case, we'll assume the robot can detect objects in a circular region around it.
        :return: List of obstacle positions detected within the sensor range
        """
        obstacles = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                distance = np.linalg.norm(np.array([x, y]) - self.robot_position)
                if distance <= self.sensor_range:
                    # Mark obstacles in the map within the sensor range
                    obstacles.append((x, y))
        return obstacles
 
    def update_map(self):
        """
        Update the map based on the sensor data.
        Mark all positions within the sensor range as obstacles (1).
        """
        obstacles = self.get_sensor_data()
        for x, y in obstacles:
            self.map[x, y] = 1  # Mark the obstacle position
 
    def move_robot(self, direction):
        """
        Move the robot in the specified direction.
        :param direction: Direction vector for movement (dx, dy)
        """
        self.robot_position += direction
        # Ensure the robot stays within the grid bounds
        self.robot_position = np.clip(self.robot_position, [0, 0], np.array(self.grid_size) - 1)
 
    def plot_map(self):
        """
        Plot the current map, showing the robot's position and obstacles.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.map, cmap='binary', origin='upper')
        plt.scatter(self.robot_position[0], self.robot_position[1], color='blue', s=100, label="Robot Position")
        plt.legend()
        plt.title("Robot Mapping System")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()
 
# 2. Initialize the robot mapping system
robot_mapping = RobotMappingSystem(grid_size=(10, 10), robot_position=(5, 5), sensor_range=3.0)
 
# 3. Simulate robot movement and map updates
for step in range(20):
    direction = np.random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])  # Random movement direction (up, down, left, right)
    robot_mapping.move_robot(direction)  # Move robot
    robot_mapping.update_map()  # Update the map based on sensor data
    robot_mapping.plot_map()  # Plot the updated map after each movement