"""
Project 672: Robot Localization System
Description:
Robot localization is the process of determining the robot's position and orientation in an environment. It is a critical component of mobile robotics, enabling robots to navigate and map their surroundings. In this project, we will implement a simple localization system using a particle filter. The particle filter will estimate the robot's position in a grid-based map by using sensor measurements (e.g., distance to obstacles) and motion data.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Particle Filter class
class ParticleFilter:
    def __init__(self, map_size, num_particles, sensor_noise=0.1):
        self.map_size = map_size
        self.num_particles = num_particles
        self.particles = np.random.rand(num_particles, 2) * map_size  # Initialize particles randomly
        self.weights = np.ones(num_particles) / num_particles  # Initialize weights equally
        self.sensor_noise = sensor_noise  # Sensor noise for measurement
 
    def move_particles(self, velocity, dt):
        """
        Move the particles based on the robot's velocity and time step.
        :param velocity: Velocity of the robot (assumed constant for simplicity)
        :param dt: Time step for simulation
        """
        self.particles += velocity * dt + np.random.randn(self.num_particles, 2) * self.sensor_noise
 
    def update_weights(self, measurement, measurement_noise):
        """
        Update the particle weights based on the sensor measurement.
        :param measurement: Sensor measurement (e.g., distance to an obstacle)
        :param measurement_noise: Noise associated with the sensor measurement
        """
        predicted_measurements = np.linalg.norm(self.particles - measurement, axis=1)
        self.weights = np.exp(-0.5 * (predicted_measurements ** 2) / (measurement_noise ** 2))
        self.weights /= np.sum(self.weights)  # Normalize weights
 
    def resample_particles(self):
        """
        Resample particles based on their weights to focus on high-probability particles.
        """
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
 
    def estimate_position(self):
        """
        Estimate the robot's position by calculating the weighted mean of the particles.
        :return: Estimated position (x, y)
        """
        return np.average(self.particles, axis=0, weights=self.weights)
 
    def plot_particles(self, estimated_position):
        """
        Visualize the particles and the estimated position.
        :param estimated_position: Estimated position of the robot
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(self.particles[:, 0], self.particles[:, 1], color='blue', s=10, label="Particles")
        plt.scatter(estimated_position[0], estimated_position[1], color='red', s=100, label="Estimated Position")
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.title("Particle Filter - Robot Localization")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
# 2. Initialize the particle filter
map_size = 10  # 10x10 grid map
num_particles = 1000  # Number of particles to use in the filter
pf = ParticleFilter(map_size, num_particles)
 
# 3. Simulate robot movement and localization
robot_position = np.array([5.0, 5.0])  # True robot position (unknown to particle filter)
velocity = np.array([0.1, 0.1])  # Constant velocity in the x and y directions
sensor_noise = 0.1  # Sensor noise for the particle filter
time_steps = 50
 
for step in range(time_steps):
    # Simulate true robot movement
    robot_position += velocity  # Move robot
    robot_position = np.clip(robot_position, 0, map_size)  # Keep robot inside map bounds
 
    # Simulate sensor measurement (distance from robot to origin, with noise)
    measurement = np.array([robot_position[0], robot_position[1]]) + np.random.randn(2) * sensor_noise
 
    # Move particles and update weights based on measurement
    pf.move_particles(velocity, 0.1)  # Move particles based on velocity
    pf.update_weights(measurement, sensor_noise)  # Update weights based on measurement
    pf.resample_particles()  # Resample particles to focus on high-probability locations
 
    # Estimate the robot's position
    estimated_position = pf.estimate_position()
 
    # Plot the particles and estimated position
    if step % 10 == 0:  # Plot every 10 steps
        pf.plot_particles(estimated_position)