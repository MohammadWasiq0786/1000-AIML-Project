"""
Project 655: Adaptive Control Implementation
Description:
Adaptive control refers to control strategies that adjust the controller parameters in real-time based on changes in the system dynamics or environment. It is particularly useful in systems with uncertain or time-varying dynamics. In this project, we will implement a simple adaptive control system that adjusts its parameters based on feedback from the system. The controller will adapt its gains to stabilize the system and achieve the desired performance.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the adaptive control system
class AdaptiveControlSystem:
    def __init__(self, desired_position=1.0, initial_gain=1.0, adaptation_rate=0.01):
        """
        Initialize the adaptive control system.
        :param desired_position: Desired position of the robot (in meters).
        :param initial_gain: Initial gain for the controller.
        :param adaptation_rate: Rate at which the controller adapts its gain.
        """
        self.desired_position = desired_position
        self.position = 0.0  # Initial position
        self.velocity = 0.0  # Initial velocity
        self.gain = initial_gain  # Controller gain
        self.adaptation_rate = adaptation_rate  # Adaptation rate for the gain
        self.history = []  # Store history of positions and velocities
 
    def compute_control(self, current_position, current_velocity, dt):
        """
        Compute the control input using adaptive control.
        :param current_position: Current position of the robot.
        :param current_velocity: Current velocity of the robot.
        :param dt: Time step for the control loop.
        :return: Control force and position updates.
        """
        # Compute the position error
        position_error = self.desired_position - current_position
 
        # Adaptive control: adjust gain based on the error
        self.gain += self.adaptation_rate * position_error  # Increase gain when error is large
 
        # Compute control force based on the position error and current velocity
        control_force = self.gain * position_error - 0.1 * current_velocity  # PD controller for adaptive control
 
        # Update position and velocity (simple model)
        self.position += current_velocity * dt
        self.velocity += control_force * dt  # Assume mass = 1 for simplicity
 
        return self.position, self.velocity, control_force
 
# 2. Initialize the adaptive control system
adaptive_control = AdaptiveControlSystem(desired_position=1.0, initial_gain=1.0, adaptation_rate=0.02)
 
# 3. Simulate the adaptive control over time
time = np.arange(0, 10, 0.1)  # Simulate for 10 seconds with a time step of 0.1s
position_history = []
velocity_history = []
control_history = []
gain_history = []
 
current_position = 0.0
current_velocity = 0.0
 
for t in time:
    # Compute the control force and update the system
    current_position, current_velocity, control_force = adaptive_control.compute_control(
        current_position, current_velocity, 0.1)  # 0.1s time step
    position_history.append(current_position)
    velocity_history.append(current_velocity)
    control_history.append(control_force)
    gain_history.append(adaptive_control.gain)
 
# 4. Plot the results of the adaptive control system
plt.figure(figsize=(10, 6))
 
# Plot position vs time
plt.subplot(3, 1, 1)
plt.plot(time, position_history, label="Position", color='blue')
plt.title('Position vs Time (Adaptive Control)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
 
# Plot velocity vs time
plt.subplot(3, 1, 2)
plt.plot(time, velocity_history, label="Velocity", color='green')
plt.title('Velocity vs Time (Adaptive Control)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
 
# Plot control force vs time
plt.subplot(3, 1, 3)
plt.plot(time, control_history, label="Control Force", color='orange')
plt.title('Control Force vs Time (Adaptive Control)')
plt.xlabel('Time (s)')
plt.ylabel('Control Force (N)')
plt.legend()
 
plt.tight_layout()
plt.show()