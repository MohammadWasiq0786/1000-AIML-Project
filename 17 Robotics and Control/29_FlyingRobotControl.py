"""
Project 669: Flying Robot Control
Description:
Flying robot control refers to the design and implementation of control systems for unmanned aerial vehicles (UAVs) or drones. These systems enable the drone to fly stably and perform tasks like navigation, obstacle avoidance, and dynamic path following. In this project, we will simulate the control of a simple quadrotor drone using PID control for position control (x, y, z) and basic stabilization. The drone will attempt to reach a target position in 3D space, considering dynamics such as acceleration and velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Drone class with PID control
class FlyingRobot:
    def __init__(self, initial_position=np.array([0.0, 0.0, 0.0]), target_position=np.array([5.0, 5.0, 5.0])):
        self.position = initial_position  # Drone's initial position (x, y, z)
        self.target_position = target_position  # Target position to reach
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
        :return: Control signal for drone (desired acceleration in x, y, z)
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
        Update the drone's position and velocity using PID control.
        :param dt: Time step for simulation
        """
        control_signal = self.pid_control()
        
        # Update the velocity and position (using simple dynamics: v = u * dt, x = x + v * dt)
        self.velocity += control_signal * dt
        self.position += self.velocity * dt
 
    def plot(self):
        """
        Visualize the drone's position on a 3D plot.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.position[0], self.position[1], self.position[2], color='blue', s=100, label="Drone Position")
        ax.scatter(self.target_position[0], self.target_position[1], self.target_position[2], color='red', s=100, label="Target Position")
        
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([0, 10])
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title("Flying Robot Control")
        ax.legend()
        plt.show()
 
# 2. Initialize the drone and simulate its movement
drone = FlyingRobot(initial_position=np.array([0.0, 0.0, 0.0]), target_position=np.array([5.0, 5.0, 5.0]))
 
# 3. Simulate the drone movement for 100 steps
time_steps = 100
for step in range(time_steps):
    drone.update_position(dt=0.1)  # 0.1s time step
    if step % 10 == 0:  # Plot every 10 steps
        drone.plot()