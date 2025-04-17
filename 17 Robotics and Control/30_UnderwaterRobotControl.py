"""
Project 670: Underwater Robot Control
Description:
Underwater robots, also known as autonomous underwater vehicles (AUVs), are designed to operate in underwater environments. These robots are used for various tasks such as exploration, mapping, and monitoring. Due to the complex and dynamic nature of underwater environments (e.g., currents, obstacles), controlling these robots can be challenging. In this project, we will simulate an underwater robot control system using a basic PID controller for depth and position control in a 2D plane (x, y, z). We will focus on maintaining the robot at a constant depth and moving it to a target position while considering forces like gravity and buoyancy.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Underwater Robot class
class UnderwaterRobot:
    def __init__(self, initial_position=np.array([0.0, 0.0, 0.0]), target_position=np.array([5.0, 5.0, 0.0])):
        self.position = initial_position  # Initial position (x, y, z)
        self.target_position = target_position  # Target position (x, y, z)
        self.velocity = np.array([0.0, 0.0, 0.0])  # Initial velocity (x, y, z)
        
        # PID controller gains
        self.kp = np.array([1.0, 1.0, 1.0])  # Proportional gain
        self.ki = np.array([0.1, 0.1, 0.1])  # Integral gain
        self.kd = np.array([0.5, 0.5, 0.5])  # Derivative gain
        
        # For the PID control
        self.error_integral = np.array([0.0, 0.0, 0.0])  # Integral of the error
        self.previous_error = np.array([0.0, 0.0, 0.0])  # Previous error for derivative term
 
    def pid_control(self):
        """
        Compute the control signal using PID control.
        :return: Control signal for underwater robot (desired acceleration in x, y, z)
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
 
    def plot(self):
        """
        Visualize the underwater robot's position in 3D space.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.position[0], self.position[1], self.position[2], color='blue', s=100, label="Robot Position")
        ax.scatter(self.target_position[0], self.target_position[1], self.target_position[2], color='red', s=100, label="Target Position")
        
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([0, 10])
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title("Underwater Robot Control")
        ax.legend()
        plt.show()
 
# 2. Initialize the underwater robot and simulate its movement
robot = UnderwaterRobot(initial_position=np.array([0.0, 0.0, 0.0]), target_position=np.array([5.0, 5.0, 0.0]))
 
# 3. Simulate the robot movement for 100 steps
time_steps = 100
for step in range(time_steps):
    robot.update_position(dt=0.1)  # 0.1s time step
    if step % 10 == 0:  # Plot every 10 steps
        robot.plot()