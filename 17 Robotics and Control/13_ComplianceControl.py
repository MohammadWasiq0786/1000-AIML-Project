"""
Project 653: Compliance Control Implementation
Description:
Compliance control in robotics refers to adjusting the robot's stiffness or compliance in response to forces exerted on it. This allows the robot to adapt to its environment, especially in tasks involving delicate interactions, such as assembly or human-robot interaction. In this project, we will implement a simple compliance control system using position and force feedback. The robot will adjust its compliance to ensure that it applies the appropriate force to the environment.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Compliance Control System
class ComplianceControlSystem:
    def __init__(self, desired_position=0.0, desired_force=0.0, stiffness=10.0, damping=0.1):
        """
        Initialize the compliance control system.
        :param desired_position: The desired position of the robot (in meters).
        :param desired_force: The desired force to be applied (in Newtons).
        :param stiffness: The stiffness of the robot (higher value means stiffer).
        :param damping: The damping factor to avoid oscillations.
        """
        self.desired_position = desired_position
        self.desired_force = desired_force
        self.stiffness = stiffness
        self.damping = damping
        self.position = 0.0  # Initial position
        self.velocity = 0.0  # Initial velocity
        self.force = 0.0  # Initial force
 
    def compute_control(self, current_position, current_velocity, current_force, dt):
        """
        Compute the control input based on position, velocity, and force feedback.
        :param current_position: The current position of the robot.
        :param current_velocity: The current velocity of the robot.
        :param current_force: The current force applied by the robot.
        :param dt: Time step for simulation.
        :return: Control force and position updates.
        """
        # Compute the position error and force error
        position_error = self.desired_position - current_position
        force_error = self.desired_force - current_force
 
        # Compute the control force based on compliance control
        control_force = self.stiffness * position_error + self.damping * current_velocity + force_error
 
        # Compute the position update (using simple dynamics for illustration)
        self.position += current_velocity * dt
        self.velocity += (control_force / 1.0) * dt  # Assume mass = 1 kg for simplicity
        self.force = control_force
 
        return self.position, self.velocity, self.force
 
# 2. Initialize the compliance control system
compliance_control = ComplianceControlSystem(desired_position=1.0, desired_force=5.0, stiffness=20.0, damping=0.5)
 
# 3. Simulate the compliance control over time
time = np.arange(0, 10, 0.1)  # Time from 0 to 10 seconds with a time step of 0.1s
position_history = []
force_history = []
velocity_history = []
 
current_position = 0.0
current_velocity = 0.0
current_force = 0.0
 
for t in time:
    current_position, current_velocity, current_force = compliance_control.compute_control(
        current_position, current_velocity, current_force, 0.1)  # 0.1s time step
    position_history.append(current_position)
    force_history.append(current_force)
    velocity_history.append(current_velocity)
 
# 4. Plot the results of the compliance control system
plt.figure(figsize=(10, 6))
 
# Plot position over time
plt.subplot(3, 1, 1)
plt.plot(time, position_history, label="Position", color='blue')
plt.axhline(compliance_control.desired_position, color='red', linestyle='--', label="Desired Position")
plt.title('Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
 
# Plot velocity over time
plt.subplot(3, 1, 2)
plt.plot(time, velocity_history, label="Velocity", color='green')
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
 
# Plot force over time
plt.subplot(3, 1, 3)
plt.plot(time, force_history, label="Force", color='orange')
plt.axhline(compliance_control.desired_force, color='red', linestyle='--', label="Desired Force")
plt.title('Force vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
 
plt.tight_layout()
plt.show()