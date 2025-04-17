"""
Project 644: Path Planning Algorithms
Description:
Path planning is the process of determining a feasible route from a start point to a goal point while avoiding obstacles. In robotics, path planning algorithms are crucial for guiding robots through environments, whether they're indoor or outdoor, to achieve a task efficiently. In this project, we will implement a path planning algorithm such as *A (A-star)**, a widely used algorithm that finds the shortest path in a grid environment with obstacles.
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the A* algorithm for path planning
class AStarPlanner:
    def __init__(self, grid, start, goal):
        self.grid = grid  # 2D grid environment (0 = free space, 1 = obstacle)
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
 
    def heuristic(self, a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
    def get_neighbors(self, node):
        # Get 4-connected neighbors (up, down, left, right)
        neighbors = []
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_row, new_col = node[0] + d[0], node[1] + d[1]
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.grid[new_row][new_col] == 0:
                neighbors.append((new_row, new_col))
        return neighbors
 
    def plan(self):
        # Initialize open and closed lists
        open_list = []
        closed_list = set()
 
        # Push the start node into the open list with a priority of 0
        heapq.heappush(open_list, (0, self.start))
 
        # Store the g and f values for each node
        g_values = {self.start: 0}
        f_values = {self.start: self.heuristic(self.start, self.goal)}
        
        # Store the parent of each node for path reconstruction
        came_from = {}
 
        while open_list:
            # Get the node with the lowest f value
            _, current = heapq.heappop(open_list)
 
            # If we reached the goal, reconstruct the path
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
 
            closed_list.add(current)
 
            # Loop through neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_list:
                    continue
 
                tentative_g_value = g_values[current] + 1  # Assuming each step costs 1
 
                if neighbor not in g_values or tentative_g_value < g_values[neighbor]:
                    came_from[neighbor] = current
                    g_values[neighbor] = tentative_g_value
                    f_values[neighbor] = tentative_g_value + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_list, (f_values[neighbor], neighbor))
 
        return []  # No path found
 
# 2. Define the grid and obstacles
grid = np.zeros((10, 10))  # 10x10 grid (0 = free, 1 = obstacle)
grid[4][4] = grid[4][5] = grid[4][6] = 1  # Adding obstacles
 
# 3. Set the start and goal positions
start = (0, 0)
goal = (9, 9)
 
# 4. Create the A* planner and find the path
planner = AStarPlanner(grid, start, goal)
path = planner.plan()
 
# 5. Visualize the grid and the path
if path:
    print(f"Path found: {path}")
    # Display grid with the path
    grid_with_path = np.copy(grid)
    for p in path:
        grid_with_path[p] = 2  # Mark the path with a 2
    plt.imshow(grid_with_path, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
else:
    print("No path found.")