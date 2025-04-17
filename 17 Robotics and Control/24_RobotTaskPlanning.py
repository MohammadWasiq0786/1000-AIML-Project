"""
Project 664: Robot Task Planning
Description:
Task planning is the process of determining a sequence of actions that a robot must take to achieve a specific goal, considering the constraints of the environment and the robot's capabilities. In this project, we will implement a simple task planning system where a robot can plan a sequence of tasks (e.g., pick an object, move it to a destination) and execute them. We will use a basic search-based planning algorithm (e.g., Breadth-First Search) to plan a series of actions in a grid environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
 
# 1. Define the grid environment and robot task planning
class RobotTaskPlanning:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.grid = np.zeros(grid_size)  # Empty grid (0 = free space, 1 = obstacle)
        self.grid[2, 2] = 1  # Adding an obstacle at position (2, 2)
        self.path = []
 
    def valid_move(self, position):
        """
        Check if a move is valid (within bounds and not an obstacle).
        :param position: Position to check (x, y)
        :return: True if valid, False otherwise
        """
        x, y = position
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and self.grid[x, y] == 0
 
    def bfs(self):
        """
        Perform Breadth-First Search (BFS) to find the shortest path from start to goal.
        :return: The path from start to goal as a list of coordinates
        """
        queue = deque([(self.start, [])])  # Queue of (current_position, path_to_here)
        visited = set([self.start])
 
        while queue:
            current_pos, path = queue.popleft()
 
            # If the goal is reached, return the path
            if current_pos == self.goal:
                return path + [current_pos]
 
            # Check the 4 possible directions (up, down, left, right)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
 
                if self.valid_move(new_pos) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [current_pos]))
 
        return []  # No path found
 
    def execute_plan(self):
        """
        Execute the planned path, visualizing each step.
        """
        self.path = self.bfs()  # Get the planned path using BFS
        if not self.path:
            print("No path found.")
            return
 
        # Simulate the robot's movement along the path
        print(f"Path found: {self.path}")
        for step in self.path:
            self.plot_grid(step)
 
    def plot_grid(self, robot_position):
        """
        Visualize the grid and the robot's position on the grid.
        """
        plt.imshow(self.grid, cmap='gray', origin='upper')
        plt.scatter(self.goal[1], self.goal[0], color='green', label="Goal", s=100)
        plt.scatter(self.start[1], self.start[0], color='blue', label="Start", s=100)
 
        # Plot the robot's position
        plt.scatter(robot_position[1], robot_position[0], color='red', label="Robot", s=100)
 
        plt.legend()
        plt.title("Robot Task Planning - BFS")
        plt.grid(True)
        plt.show()
 
# 2. Initialize the robot task planning system
task_planner = RobotTaskPlanning(start=(0, 0), goal=(4, 4))
 
# 3. Execute the task planning
task_planner.execute_plan()