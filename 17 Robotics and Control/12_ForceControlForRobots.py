"""
Project 652: Force Control for Robots
Description:
Force control in robotics is used to control the interaction forces between a robot and its environment, such as when a robot manipulates objects or performs tasks like assembly or tactile exploration. The goal is to ensure that the robot applies the correct amount of force, preventing damage to the object or environment while maintaining efficient task performance. In this project, we will implement a simple force control system where the robot adjusts its force based on feedback from force sensors.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the robot and force control system
class ForceControlSystem:
    def __init__(self, desired_force=5.0, max_force=10.0, kp=0.1, ki=0.05, kd=0.01):
        """
        Initialize the force control system.
        :param desired_force: The desired force to apply (in Newtons).
        :param max_force: The maximum allowable force.
        :param kp, ki, kd: PID controller gains.
        """
        self.desired_force = desired_force
        self.max_force = max_force
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0
        self.integral = 0
 
    def compute_control(self, measured_force, dt):
        """
        Compute the control signal using a PID controller.
        :param measured_force: The measured force (in Newtons).
        :param dt: The time step for the control loop.
        :return: The control force to be applied.
        """
        error = self.desired_force - measured_force
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
 
        # PID control
        control_force = self.kp * error + self.ki * self.integral + self.kd * derivative
        control_force = np.clip(control_force, 0, self.max_force)  # Ensure the force is within limits
 
        self.prev_error = error
        return control_force
 
# 2. Simulate a robot interaction with an object
class RobotInteraction:
    def __init__(self):
        self.measured_force = 0  # Initial force is 0
        self.velocity = 0  # Initial velocity is 0
        self.time_step = 0.1  # Time step for the simulation (in seconds)
 
    def apply_force(self, control_force):
        """
        Apply the control force to the object and update the measured force.
        :param control_force: The force to be applied to the object (in Newtons).
        """
        # Simulate the interaction: the force applied is proportional to the control force
        self.measured_force = control_force
        # Update velocity based on the force (simple model)
        self.velocity = self.measured_force * 0.1  # Assume some mass for the object (arbitrary value)
    
    def get_measured_force(self):
        return self.measured_force
 
# 3. Initialize the force control system and robot interaction
force_control_system = ForceControlSystem(desired_force=5.0)
robot_interaction = RobotInteraction()
 
# 4. Simulate the force control over time
time = np.arange(0, 10, robot_interaction.time_step)
force_history = []
 
for t in time:
    # Apply the control force based on the measured force
    measured_force = robot_interaction.get_measured_force()
    control_force = force_control_system.compute_control(measured_force, robot_interaction.time_step)
    robot_interaction.apply_force(control_force)
    
    # Record the force for visualization
    force_history.append(measured_force)
 
# 5. Plot the force over time
plt.figure(figsize=(10, 6))
plt.plot(time, force_history, label="Measured Force", color='blue')
plt.axhline(force_control_system.desired_force, color='red', linestyle='--', label="Desired Force")
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force Control for Robot Interaction')
plt.legend()
plt.grid(True)
plt.show()