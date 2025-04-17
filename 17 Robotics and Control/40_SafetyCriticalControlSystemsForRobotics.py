"""
Project 680: Safety-Critical Control Systems for Robotics
Description:
Safety-critical control systems are essential in robotics, particularly for systems that operate in environments where failures could result in significant harm, such as autonomous vehicles, industrial robots, or medical robots. In this project, we will implement a safety-critical control system for a robot that incorporates safety mechanisms, such as emergency stop or collision avoidance. The robot will move towards a target while continuously checking for safety constraints (e.g., obstacles or out-of-bound conditions) and applying safety measures when necessary.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Safety-Critical Robot class
class SafetyCriticalRobot:
    def __init__(self, grid_size=(10, 10), start_position=(0, 0), goal_position=(9, 9), safety_radius=1.0):
        self.grid_size = grid_size  # Size of the grid environment
        self.position = np.array(start_position)  # Initial robot position
        self.goal_position = np.array(goal_position)  # Goal position
        self.safety_radius = safety_radius  # Safety radius to avoid obstacles or boundaries
        self.velocity = np.array([0.1, 0.1])  # Initial velocity (x, y)
        self.obstacles = self.generate_obstacles()  # Generate obstacles
        self.history = []  # To store the history of robot positions for visualization
 
    def generate_obstacles(self):
        """
        Randomly generate obstacles within the grid that the robot must avoid.
        :return: List of obstacle positions
        """
        obstacles = []
        for _ in range(5):
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            obstacles.append(np.array([x, y]))
        return obstacles
 
    def check_safety(self):
        """
        Check if the robot is within a safety radius of any obstacles or boundaries.
        :return: True if the robot is in a dangerous position, False otherwise
        """
        # Check proximity to obstacles
        for obstacle in self.obstacles:
            if np.linalg.norm(self.position - obstacle) < self.safety_radius:
                return True  # Robot is too close to an obstacle
 
        # Check proximity to grid boundaries
        if np.any(self.position < 0) or np.any(self.position >= self.grid_size):
            return True  # Robot is out of bounds
 
        return False
 
    def move(self):
        """
        Move the robot towards the goal with safety checks.
        If the robot is too close to an obstacle or boundary, it stops or changes direction.
        """
        if self.check_safety():
            print("Safety alert! Robot is too close to an obstacle or boundary. Stopping.")
            return  # Stop if the robot is too close to an obstacle or out of bounds
 
        # Simple movement towards the goal
        direction = self.goal_position - self.position
        distance = np.linalg.norm(direction)
 
        if distance > 0:
            # Normalize the direction vector
            direction = direction / distance
            self.position += direction * self.velocity  # Move robot
 
    def plot_environment(self):
        """
        Visualize the environment, including obstacles, robot position, and goal.
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(self.goal_position[0], self.goal_position[1], color='red', s=100, label="Goal Position")
        plt.scatter(self.position[0], self.position[1], color='blue', s=100, label="Robot Position")
 
        # Plot obstacles
        for obstacle in self.obstacles:
            plt.scatter(obstacle[0], obstacle[1], color='black', s=100, label="Obstacle")
 
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.title("Safety-Critical Robot Control System")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
    def navigate(self):
        """
        Simulate the robot navigating the environment with safety constraints.
        """
        while np.linalg.norm(self.position - self.goal_position) > 0.1:  # Until the robot reaches the goal
            self.move()  # Move the robot
            self.plot_environment()  # Visualize the robot's movement
 
# 2. Initialize the safety-critical robot
robot_safety = SafetyCriticalRobot(grid_size=(10, 10), start_position=(0, 0), goal_position=(9, 9), safety_radius=1.0)
 
# 3. Simulate the robot navigation with safety-critical control
robot_safety.navigate()

