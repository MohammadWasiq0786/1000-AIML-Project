"""
Project 671: Mobile Robot Navigation
Description:
Mobile robot navigation involves planning and controlling the movement of a robot in an environment to reach a goal while avoiding obstacles. The robot must be able to localize itself within the environment, plan a path to the goal, and control its motion to avoid collisions. In this project, we will simulate a mobile robot navigation system using A path planning* for navigation and PID control for movement. The robot will navigate through a grid with obstacles and move towards a goal while avoiding collisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq
 
# 1. Define the A* path planning class
class AStarPathPlanner:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
 
    def heuristic(self, node):
        """
        Heuristic function: Manhattan distance
        :param node: Current node (x, y)
        :return: Manhattan distance to the goal
        """
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])
 
    def get_neighbors(self, node):
        """
        Get valid neighbors for the current node.
        :param node: Current node (x, y)
        :return: List of valid neighbors
        """
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx][ny] == 0:
                neighbors.append((nx, ny))
        return neighbors
 
    def plan_path(self):
        """
        Plan a path from the start to the goal using A* algorithm.
        :return: List of nodes representing the planned path
        """
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(self.start), 0, self.start, None))  # f, g, node, parent
        closed_list = set()
        came_from = {}
 
        while open_list:
            _, g, current, parent = heapq.heappop(open_list)
 
            if current == self.goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]  # Return reversed path
 
            closed_list.add(current)
            came_from[current] = parent
 
            for neighbor in self.get_neighbors(current):
                if neighbor not in closed_list:
                    heapq.heappush(open_list, (g + 1 + self.heuristic(neighbor), g + 1, neighbor, current))
 
        return []  # No path found
 
# 2. Define the Mobile Robot class with PID control
class MobileRobot:
    def __init__(self, start_position=(0, 0), target_position=(5, 5), grid_size=(10, 10)):
        self.position = np.array(start_position)  # Initial position (x, y)
        self.target_position = np.array(target_position)  # Target position (x, y)
        self.velocity = np.array([0.0, 0.0])  # Initial velocity (x, y)
        self.grid_size = grid_size
        self.path = []  # Path planned by A* planner
        self.current_goal_index = 0  # Index of the current goal on the path
 
        # PID controller gains
        self.kp = np.array([1.0, 1.0])  # Proportional gain
        self.ki = np.array([0.1, 0.1])  # Integral gain
        self.kd = np.array([0.5, 0.5])  # Derivative gain
        
        self.error_integral = np.array([0.0, 0.0])  # Integral of the error
        self.previous_error = np.array([0.0, 0.0])  # Previous error for derivative term
 
    def pid_control(self):
        """
        Compute the control signal using PID control.
        :return: Control signal for robot movement
        """
        # Calculate the error (difference between target and current position)
        error = self.target_position - self.position
        self.error_integral += error  # Integrate the error over time
        error_derivative = error - self.previous_error  # Derivative of the error
        
        # PID control law
        control_signal = (self.kp * error + self.ki * self.error_integral + self.kd * error_derivative)
        
        # Update the previous error for the next iteration
        self.previous_error = error
        
        return control_signal
 
    def update_position(self, dt):
        """
        Update the robot's position and velocity using PID control.
        :param dt: Time step for simulation
        """
        control_signal = self.pid_control()
        
        # Update the velocity and position (using simple dynamics: v = u * dt, x = x + v * dt)
        self.velocity += control_signal * dt
        self.position += self.velocity * dt
 
    def set_target_position(self, target_position):
        """
        Set the target position for the robot and plan the path using A*.
        :param target_position: New target position
        """
        self.target_position = np.array(target_position)
        planner = AStarPathPlanner(grid=self.grid, start=self.position, goal=self.target_position)
        self.path = planner.plan_path()
        self.current_goal_index = 0
 
    def plot(self):
        """
        Visualize the robot's movement on a 2D grid.
        """
        plt.figure(figsize=(8, 8))
        plt.plot(self.position[0], self.position[1], 'bo', label="Robot Position")
        plt.plot(self.target_position[0], self.target_position[1], 'ro', label="Target Position")
 
        if self.path:
            path_x, path_y = zip(*self.path)
            plt.plot(path_x, path_y, 'g--', label="Planned Path")
 
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.title("Mobile Robot Navigation with A* and PID")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
# 3. Define the grid environment and set the initial position and target position
grid = np.zeros((10, 10))  # 10x10 grid with no obstacles
grid[3:7, 3:7] = 1  # Adding a simple obstacle (4x4 block in the middle)
 
# 4. Initialize the mobile robot and simulate navigation
robot = MobileRobot(start_position=(0, 0), target_position=(8, 8), grid_size=(10, 10))
 
# 5. Plan the path using A* and move the robot towards the target
for step in range(50):
    if robot.path and robot.current_goal_index < len(robot.path):
        robot.set_target_position(robot.path[robot.current_goal_index])
        robot.update_position(dt=0.1)  # Move robot for 0.1s time step
        robot.current_goal_index += 1
        if step % 10 == 0:  # Plot every 10 steps
            robot.plot()