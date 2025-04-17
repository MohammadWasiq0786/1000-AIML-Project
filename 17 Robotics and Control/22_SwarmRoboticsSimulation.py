"""
Project 662: Swarm Robotics Simulation
Description:
Swarm robotics is a field in which a large number of simple robots work together to complete a task through decentralized control. The robots typically follow simple rules and interact with each other, resulting in emergent behaviors. In this project, we will simulate a swarm of robots using basic rules of interaction, where robots will explore the environment and attempt to cover a target area while avoiding collisions. We will implement a simple flocking behavior where robots move in a coordinated way based on local interactions with their neighbors.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot class with flocking behavior
class Robot:
    def __init__(self, position, velocity):
        self.position = np.array(position)  # Initial position
        self.velocity = np.array(velocity)  # Initial velocity
 
    def update_position(self, dt):
        # Update robot's position based on its velocity
        self.position += self.velocity * dt
 
    def apply_flocking(self, robots, perception_range=1.0):
        """
        Apply flocking behavior: Each robot adjusts its velocity based on its neighbors.
        :param robots: List of all robots in the swarm
        :param perception_range: Range within which robots can interact
        """
        alignment = np.zeros(2)
        cohesion = np.zeros(2)
        separation = np.zeros(2)
        count = 0
        
        for robot in robots:
            if np.linalg.norm(self.position - robot.position) < perception_range:
                # Alignment: Move in the same direction as neighbors
                alignment += robot.velocity
 
                # Cohesion: Move towards the center of mass of neighbors
                cohesion += robot.position
 
                # Separation: Avoid collisions with neighbors
                separation += self.position - robot.position
 
                count += 1
 
        if count > 0:
            # Normalize vectors
            alignment /= count
            cohesion /= count
            separation /= count
 
            # Apply simple rules: Weighted sum of behaviors
            self.velocity += 0.1 * alignment + 0.1 * cohesion + 0.2 * separation
            self.velocity = self.velocity / np.linalg.norm(self.velocity)  # Normalize velocity
 
# 2. Define the Swarm class
class Swarm:
    def __init__(self, num_robots, area_size=(10, 10), initial_velocity=(0.1, 0.1)):
        self.num_robots = num_robots
        self.robots = []
        self.area_size = area_size
 
        # Initialize robots with random positions and velocities
        for _ in range(num_robots):
            position = np.random.uniform(0, area_size[0], 2)
            velocity = np.random.uniform(-0.1, 0.1, 2)
            self.robots.append(Robot(position, velocity))
 
    def update(self, dt):
        # Update the position and velocity of each robot based on flocking behavior
        for robot in self.robots:
            robot.apply_flocking(self.robots)
            robot.update_position(dt)
 
    def plot(self):
        # Plot the positions of the robots
        plt.figure(figsize=(8, 8))
        for robot in self.robots:
            plt.scatter(robot.position[0], robot.position[1], color='blue')
        plt.xlim(0, self.area_size[0])
        plt.ylim(0, self.area_size[1])
        plt.title("Swarm Robotics Simulation with Flocking Behavior")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()
 
# 3. Initialize the swarm with 10 robots and simulate their behavior
swarm = Swarm(num_robots=10)
 
# 4. Simulate the swarm movement for 100 time steps
for step in range(100):
    swarm.update(dt=0.1)  # Update the swarm every 0.1 seconds
    if step % 10 == 0:  # Plot every 10 steps
        swarm.plot()