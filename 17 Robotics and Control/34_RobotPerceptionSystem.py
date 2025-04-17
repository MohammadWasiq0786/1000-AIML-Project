"""
Project 674: Robot Perception System
Description:
A robot perception system is responsible for processing sensory data and interpreting the environment, enabling the robot to understand and act accordingly. This can include tasks like object detection, obstacle avoidance, and environment mapping. In this project, we will implement a basic robot perception system that uses a sensor array (e.g., distance sensors or a camera) to detect objects and obstacles in its environment. The system will be capable of performing object detection and making decisions based on the perceived environment.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot Perception System class
class RobotPerceptionSystem:
    def __init__(self, grid_size=(10, 10), robot_position=(5, 5), sensor_range=3.0):
        self.grid_size = grid_size  # Size of the environment grid
        self.robot_position = np.array(robot_position)  # Initial position of the robot
        self.sensor_range = sensor_range  # Range of the robot's sensor
        self.map = np.zeros(grid_size)  # Empty environment map (0 = free space, 1 = obstacle)
        self.history = []  # To store history of positions for visualization
 
    def generate_random_obstacles(self, num_obstacles=5):
        """
        Randomly generate obstacles within the environment (map).
        :param num_obstacles: Number of obstacles to generate
        """
        for _ in range(num_obstacles):
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            self.map[x, y] = 1  # Mark an obstacle on the map
 
    def get_sensor_data(self):
        """
        Simulate sensor data by detecting obstacles within the sensor range.
        :return: List of obstacle positions detected within the sensor range
        """
        detected_obstacles = []
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                distance = np.linalg.norm(np.array([x, y]) - self.robot_position)
                if distance <= self.sensor_range and self.map[x, y] == 1:
                    detected_obstacles.append((x, y))  # Detected obstacle
        return detected_obstacles
 
    def plot_perception(self, detected_obstacles):
        """
        Visualize the robot's perception of the environment, including obstacles.
        :param detected_obstacles: List of detected obstacles within the sensor range
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.map, cmap='binary', origin='upper', extent=(0, self.grid_size[0], 0, self.grid_size[1]))
        plt.scatter(self.robot_position[0], self.robot_position[1], color='blue', s=100, label="Robot Position")
        
        # Mark detected obstacles
        if detected_obstacles:
            for obs in detected_obstacles:
                plt.scatter(obs[0], obs[1], color='red', s=100, label="Obstacle")
 
        plt.legend()
        plt.title("Robot Perception System")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()
 
# 2. Initialize the robot perception system and generate obstacles
robot_perception = RobotPerceptionSystem(grid_size=(10, 10), robot_position=(5, 5), sensor_range=3.0)
robot_perception.generate_random_obstacles(num_obstacles=5)  # Generate 5 random obstacles
 
# 3. Simulate the perception of the robot (detecting obstacles in the environment)
detected_obstacles = robot_perception.get_sensor_data()  # Get sensor data (detected obstacles)
 
# 4. Visualize the robot's perception of the environment
robot_perception.plot_perception(detected_obstacles)