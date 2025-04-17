"""
Project 679: Sim-to-Real Transfer for Robotics
Description:
Sim-to-real transfer refers to the process of transferring a robot's learned behavior from a simulated environment to the real world. A key challenge in robotics is that models trained in simulation often do not perform well when deployed in real-world environments due to discrepancies between the simulation and reality (known as the reality gap). In this project, we will simulate a robot's movement in a controlled environment and then attempt to "transfer" this behavior to a real-world-like setup by applying simple adjustments to handle discrepancies, such as friction or noise.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot class with Sim-to-Real Transfer capabilities
class SimToRealRobot:
    def __init__(self, grid_size=(10, 10), start_position=(0, 0), goal_position=(9, 9)):
        self.grid_size = grid_size  # Size of the grid environment
        self.position = np.array(start_position)  # Initial position of the robot
        self.goal_position = np.array(goal_position)  # Goal position
        self.velocity = np.array([0.1, 0.1])  # Initial velocity (x, y)
        self.simulation_noise = 0.05  # Noise added to simulation behavior (sim-to-real gap)
 
    def move_robot(self):
        """
        Move the robot towards the goal with some added noise for sim-to-real transfer.
        Simulating the discrepancy between the simulated environment and real-world conditions.
        """
        direction = self.goal_position - self.position
        distance = np.linalg.norm(direction)
 
        if distance > 0:
            # Normalize the direction vector
            direction = direction / distance
            noise = np.random.randn(2) * self.simulation_noise  # Simulate real-world noise
            self.position += (direction * self.velocity + noise)  # Move robot with noise
 
    def plot_environment(self):
        """
        Visualize the robot's position and the target position.
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(self.position[0], self.position[1], color='blue', s=100, label="Robot Position")
        plt.scatter(self.goal_position[0], self.goal_position[1], color='red', s=100, label="Goal Position")
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.title("Sim-to-Real Transfer: Robot Navigation")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
    def simulate_navigation(self):
        """
        Simulate robot navigation with the Sim-to-Real Transfer approach.
        The robot will attempt to navigate toward the goal while considering the real-world noise.
        """
        while np.linalg.norm(self.position - self.goal_position) > 0.1:  # Stop when close to goal
            self.move_robot()
            self.plot_environment()  # Visualize the robot's position during navigation
 
# 2. Initialize the Sim-to-Real robot
robot_sim_to_real = SimToRealRobot(grid_size=(10, 10), start_position=(0, 0), goal_position=(9, 9))
 
# 3. Simulate the robot navigation with noise to mimic the real-world discrepancy
robot_sim_to_real.simulate_navigation()