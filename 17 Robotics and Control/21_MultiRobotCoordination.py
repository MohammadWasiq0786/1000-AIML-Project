"""
Project 661: Multi-Robot Coordination
Description:
Multi-robot coordination involves multiple robots working together to accomplish a task, such as searching an area, transporting objects, or exploring an environment. The coordination can involve path planning, task assignment, and communication between robots. In this project, we will simulate a simple multi-robot system where robots coordinate to reach designated target locations without colliding with each other. We will implement a basic centralized coordination strategy, where a central controller assigns tasks and coordinates the movements of the robots.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot class
class Robot:
    def __init__(self, position, target):
        self.position = np.array(position)  # Initial position
        self.target = np.array(target)  # Target position
        self.velocity = 0.1  # Robot speed
 
    def move_towards_target(self):
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        
        # Normalize the direction and move the robot
        if distance > 0:
            direction = direction / distance
            self.position += direction * self.velocity  # Move robot towards target
 
# 2. Create the multi-robot coordination system
class MultiRobotCoordination:
    def __init__(self, num_robots, area_size=(10, 10)):
        self.num_robots = num_robots
        self.area_size = area_size  # Define the area for the robots to move
        self.robots = []
        self.targets = []
 
        # Initialize robots and target positions
        for i in range(num_robots):
            robot_start = np.random.uniform(0, area_size[0], 2)  # Random start position within the area
            robot_target = np.random.uniform(0, area_size[0], 2)  # Random target position within the area
            robot = Robot(robot_start, robot_target)
            self.robots.append(robot)
            self.targets.append(robot_target)
 
    def update(self):
        # Update the robots' positions towards their targets
        for robot in self.robots:
            robot.move_towards_target()
 
    def check_collisions(self):
        # Check if any robots are too close to each other (collision avoidance)
        for i in range(len(self.robots)):
            for j in range(i + 1, len(self.robots)):
                if np.linalg.norm(self.robots[i].position - self.robots[j].position) < 0.5:
                    return True  # Collision detected
        return False
 
    def plot(self):
        # Plot the robots and their targets
        plt.figure(figsize=(8, 8))
        for i in range(self.num_robots):
            plt.scatter(self.robots[i].position[0], self.robots[i].position[1], color='blue', label=f"Robot {i+1}")
            plt.scatter(self.targets[i][0], self.targets[i][1], color='red', marker='x', label=f"Target {i+1}")
        plt.xlim(0, self.area_size[0])
        plt.ylim(0, self.area_size[1])
        plt.title("Multi-Robot Coordination")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.legend()
        plt.show()
 
# 3. Initialize the multi-robot system and simulate the coordination
num_robots = 5  # Number of robots in the system
coordination_system = MultiRobotCoordination(num_robots)
 
# 4. Simulate the robots' movement and coordination
num_steps = 100
for step in range(num_steps):
    if coordination_system.check_collisions():
        print("Collision detected!")
        break
    coordination_system.update()
 
    if step % 10 == 0:
        coordination_system.plot()  # Plot the robots and their targets at every 10th step