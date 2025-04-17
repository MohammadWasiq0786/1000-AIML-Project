"""
Project 645: Obstacle Avoidance System
Description:
An obstacle avoidance system is a critical part of robotic navigation, where a robot must detect obstacles in its path and take appropriate action to avoid collisions. This system uses sensors, such as cameras or LiDAR, to perceive the environment and uses algorithms to make decisions about how to navigate safely. In this project, we will implement a simple obstacle avoidance system for a robot using a reactive control approach. The robot will detect obstacles in its environment and change its trajectory to avoid them.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the robot and environment
class Robot:
    def __init__(self, position, velocity=0.1):
        self.position = np.array(position)  # Position (x, y)
        self.velocity = velocity  # Robot velocity
 
    def move(self, direction):
        # Move the robot in a specific direction (angle in radians)
        self.position += self.velocity * np.array([np.cos(direction), np.sin(direction)])
 
# 2. Define the environment with obstacles
class ObstacleAvoidanceEnv:
    def __init__(self, robot, obstacles, goal):
        self.robot = robot
        self.obstacles = obstacles  # List of obstacles as (x, y) coordinates
        self.goal = goal  # Goal position
        self.robot_radius = 0.1  # Radius of the robot for collision detection
 
    def detect_obstacle(self):
        # Check if the robot is near any obstacle
        for obstacle in self.obstacles:
            if np.linalg.norm(self.robot.position - obstacle) < self.robot_radius:
                return True
        return False
 
    def is_goal_reached(self):
        # Check if the robot has reached the goal
        return np.linalg.norm(self.robot.position - self.goal) < self.robot_radius
 
    def move_robot(self, direction):
        # Move the robot and check for collisions
        self.robot.move(direction)
        return self.detect_obstacle()
 
# 3. Define the obstacle avoidance control logic
def obstacle_avoidance(env):
    # Simple reactive control: if an obstacle is detected, change direction
    direction = 0  # Start moving towards the goal
    while not env.is_goal_reached():
        if env.detect_obstacle():
            # If an obstacle is detected, change direction randomly
            direction += np.pi / 4  # Turn 45 degrees to avoid the obstacle
        else:
            # Continue moving in the current direction towards the goal
            direction = np.arctan2(env.goal[1] - env.robot.position[1], env.goal[0] - env.robot.position[0])
        
        # Move the robot
        env.move_robot(direction)
        plot_robot_and_obstacles(env)
 
# 4. Plot the robot and obstacles
def plot_robot_and_obstacles(env):
    plt.clf()
    plt.plot(env.goal[0], env.goal[1], 'go', label="Goal")
    plt.plot(env.robot.position[0], env.robot.position[1], 'bo', label="Robot")
    for obstacle in env.obstacles:
        plt.plot(obstacle[0], obstacle[1], 'ro', label="Obstacle")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.pause(0.1)
 
# 5. Initialize the robot, obstacles, and goal
robot = Robot(position=[-1, -1])
obstacles = [(0, 0), (0.5, 0.5), (-0.5, 0.5)]  # Example obstacles
goal = [1, 1]  # Goal position
 
# 6. Initialize the environment and run the obstacle avoidance system
env = ObstacleAvoidanceEnv(robot, obstacles, goal)
plt.figure()
obstacle_avoidance(env)
plt.show()