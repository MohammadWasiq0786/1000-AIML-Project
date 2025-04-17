"""
Project 678: Robot Simulation Environment
Description:
A robot simulation environment is a virtual platform that mimics the physical world where robots can be tested, trained, and optimized. The simulation allows robots to interact with obstacles, perform tasks, and learn without the need for real-world hardware. In this project, we will create a basic robot simulation environment where a robot can navigate a grid world with obstacles. The robot will use a simple navigation algorithm to move towards a goal while avoiding obstacles.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot Simulation Environment class
class RobotSimulationEnvironment:
    def __init__(self, grid_size=(10, 10), start_position=(0, 0), goal_position=(9, 9), num_obstacles=10):
        self.grid_size = grid_size  # Size of the environment grid
        self.start_position = np.array(start_position)  # Initial robot position
        self.goal_position = np.array(goal_position)  # Goal position
        self.robot_position = self.start_position  # Robot's current position
        self.num_obstacles = num_obstacles  # Number of obstacles in the environment
        self.grid = np.zeros(grid_size)  # Empty grid (0 = free space, 1 = obstacle)
        self.generate_obstacles()  # Generate obstacles in the grid
        self.history = []  # To store the history of robot positions for visualization
 
    def generate_obstacles(self):
        """
        Randomly generate obstacles in the grid.
        """
        for _ in range(self.num_obstacles):
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            # Avoid placing obstacles on the start or goal positions
            if (x, y) != tuple(self.start_position) and (x, y) != tuple(self.goal_position):
                self.grid[x, y] = 1  # Mark the position as an obstacle
 
    def move_robot(self, direction):
        """
        Move the robot in the specified direction, considering obstacles and boundaries.
        :param direction: Direction to move the robot (up, down, left, right)
        """
        x, y = self.robot_position
        if direction == 'up' and x > 0 and self.grid[x - 1, y] != 1:
            self.robot_position = np.array([x - 1, y])
        elif direction == 'down' and x < self.grid_size[0] - 1 and self.grid[x + 1, y] != 1:
            self.robot_position = np.array([x + 1, y])
        elif direction == 'left' and y > 0 and self.grid[x, y - 1] != 1:
            self.robot_position = np.array([x, y - 1])
        elif direction == 'right' and y < self.grid_size[1] - 1 and self.grid[x, y + 1] != 1:
            self.robot_position = np.array([x, y + 1])
 
    def plot_environment(self):
        """
        Visualize the simulation environment, including obstacles and robot position.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='binary', origin='upper')  # Plot grid with obstacles
        plt.scatter(self.goal_position[1], self.goal_position[0], color='green', s=100, label="Goal Position")
        plt.scatter(self.robot_position[1], self.robot_position[0], color='blue', s=100, label="Robot Position")
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.title("Robot Simulation Environment")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
    def simulate_navigation(self):
        """
        Simulate robot navigation to reach the goal.
        The robot will try to move toward the goal while avoiding obstacles.
        """
        directions = ['up', 'down', 'left', 'right']
        
        # Simulate robot movement until it reaches the goal
        while not np.array_equal(self.robot_position, self.goal_position):
            # Naive approach: Move towards the goal (simple strategy for demo purposes)
            if self.robot_position[0] < self.goal_position[0]:
                direction = 'down'
            elif self.robot_position[0] > self.goal_position[0]:
                direction = 'up'
            elif self.robot_position[1] < self.goal_position[1]:
                direction = 'right'
            else:
                direction = 'left'
            
            # Move the robot in the chosen direction
            self.move_robot(direction)
            self.history.append(self.robot_position.copy())  # Store the robot's position
 
            # Plot the environment every 10 steps
            if len(self.history) % 10 == 0:
                self.plot_environment()
 
# 2. Initialize the robot simulation environment
robot_simulation = RobotSimulationEnvironment(grid_size=(10, 10), start_position=(0, 0), goal_position=(9, 9), num_obstacles=15)
 
# 3. Simulate the robot navigation to reach the goal
robot_simulation.simulate_navigation()