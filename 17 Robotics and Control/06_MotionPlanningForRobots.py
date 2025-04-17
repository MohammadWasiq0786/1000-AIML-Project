"""
Project 646: Motion Planning for Robots
Description:
Motion planning is the process of determining a feasible path for a robot to follow in order to move from a start position to a goal position while avoiding obstacles. This is a fundamental task in robotics, especially for mobile robots. In this project, we will implement a motion planning algorithm such as Rapidly-exploring Random Trees (RRT) to plan a path for a robot in a 2D environment. The robot will navigate through a simple grid-based environment with obstacles, finding a path to its goal.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
 
# 1. Define the RRT algorithm for motion planning
class RRT:
    def __init__(self, start, goal, obstacles, bounds, max_iter=1000, step_size=0.1):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree = [self.start]  # Initialize tree with the start position
 
    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
 
    def nearest_node(self, node):
        # Find the nearest node in the tree to the given node
        return min(self.tree, key=lambda n: self.distance(n, node))
 
    def is_collision_free(self, p1, p2):
        # Check if the line segment between p1 and p2 intersects any obstacle
        for obs in self.obstacles:
            if self.distance(p1, obs) < self.step_size or self.distance(p2, obs) < self.step_size:
                return False
        return True
 
    def step_towards(self, p1, p2):
        # Move from p1 towards p2 by a step size
        direction = (p2 - p1) / self.distance(p1, p2)
        return p1 + self.step_size * direction
 
    def plan(self):
        for _ in range(self.max_iter):
            random_node = np.array([random.uniform(self.bounds[0], self.bounds[2]),
                                    random.uniform(self.bounds[1], self.bounds[3])])
            nearest = self.nearest_node(random_node)
            new_node = self.step_towards(nearest, random_node)
            if self.is_collision_free(nearest, new_node):
                self.tree.append(new_node)
                if self.distance(new_node, self.goal) < self.step_size:
                    return self.reconstruct_path(new_node)
        return []
 
    def reconstruct_path(self, node):
        # Reconstruct the path from the goal to the start
        path = [node]
        while not np.array_equal(path[-1], self.start):
            nearest = self.nearest_node(path[-1])
            path.append(nearest)
        path.reverse()
        return path
 
# 2. Define the environment for motion planning
obstacles = [(0.5, 0.5), (1.5, 1.5), (2, 1)]  # Example obstacles
start = (0, 0)  # Start position
goal = (2, 2)  # Goal position
bounds = [0, 0, 3, 3]  # Bounds for the environment (x_min, y_min, x_max, y_max)
 
# 3. Initialize the RRT planner and plan the motion
planner = RRT(start=start, goal=goal, obstacles=obstacles, bounds=bounds)
path = planner.plan()
 
# 4. Plot the environment and the planned path
plt.figure(figsize=(6, 6))
for obs in obstacles:
    plt.scatter(obs[0], obs[1], color='red', s=100, label="Obstacle")
plt.scatter(start[0], start[1], color='green', s=100, label="Start")
plt.scatter(goal[0], goal[1], color='blue', s=100, label="Goal")
 
if path:
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color='purple', lw=2, label="Path")
 
plt.xlim(bounds[0], bounds[2])
plt.ylim(bounds[1], bounds[3])
plt.legend()
plt.title("RRT Path Planning")
plt.grid(True)
plt.show()