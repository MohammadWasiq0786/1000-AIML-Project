"""
Project 657: Nonlinear Control Systems
Description:
Nonlinear control systems deal with systems where the control law is not a linear function of the system’s states or control inputs. Many real-world systems, especially robotic systems, exhibit nonlinear behavior. In this project, we will implement a nonlinear control system for a simple robotic system. We will apply a backstepping control method, a popular approach for stabilizing nonlinear systems, to control the robot’s position and velocity.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the system dynamics (a simple second-order nonlinear system)
def nonlinear_system(x, u):
    """
    Nonlinear system dynamics: x' = Ax + Bu + f(x)
    :param x: State vector (position, velocity)
    :param u: Control input (force)
    :return: dx (derivative of state)
    """
    position, velocity = x
    f = -0.1 * velocity**3  # Nonlinear damping term (example)
    A = np.array([[0, 1], [0, -0.1]])  # System dynamics matrix (linear part)
    B = np.array([0, 1])  # Control input matrix (force)
    dx = A @ x + B * u + np.array([0, f])  # State derivative (position, velocity)
    return dx
 
# 2. Define the backstepping control law
class NonlinearControl:
    def __init__(self, desired_position=1.0, desired_velocity=0.0):
        """
        Initialize the nonlinear control system.
        :param desired_position: Desired position of the robot.
        :param desired_velocity: Desired velocity of the robot.
        """
        self.desired_position = desired_position
        self.desired_velocity = desired_velocity
        self.position = 0.0  # Initial position
        self.velocity = 0.0  # Initial velocity
        self.state = np.array([self.position, self.velocity])
        self.alpha = 1.0  # Backstepping control gain for position
        self.beta = 1.0  # Backstepping control gain for velocity
 
    def backstepping_control(self, dt):
        """
        Apply the backstepping control law to the nonlinear system.
        :param dt: Time step for simulation.
        :return: Control input (force)
        """
        position, velocity = self.state
        # Backstepping design
        e1 = self.desired_position - position  # Position error
        e2 = self.desired_velocity - velocity  # Velocity error
 
        # Control law for position and velocity
        v1 = self.alpha * e1 + self.beta * e2
        u = self.alpha * v1 + self.beta * e2 + 0.1 * velocity**2  # Adding nonlinear term (example)
        
        # Apply the control input and update state using system dynamics
        dx = nonlinear_system(self.state, u)
        self.state = self.state + dx * dt  # Update state using Euler integration
        return u, self.state
 
# 3. Initialize the nonlinear control system
control_system = NonlinearControl(desired_position=1.0, desired_velocity=0.0)
 
# 4. Simulate the nonlinear control system over time
time = np.arange(0, 10, 0.1)  # Simulate for 10 seconds with a time step of 0.1s
position_history = []
velocity_history = []
control_history = []
 
for t in time:
    u, state = control_system.backstepping_control(dt=0.1)  # 0.1s time step
    position_history.append(state[0])
    velocity_history.append(state[1])
    control_history.append(u)
 
# 5. Plot the results of the nonlinear control system
plt.figure(figsize=(10, 6))
 
# Plot position vs time
plt.subplot(3, 1, 1)
plt.plot(time, position_history, label="Position", color='blue')
plt.title('Position vs Time (Nonlinear Control)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
 
# Plot velocity vs time
plt.subplot(3, 1, 2)
plt.plot(time, velocity_history, label="Velocity", color='green')
plt.title('Velocity vs Time (Nonlinear Control)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
 
# Plot control input vs time
plt.subplot(3, 1, 3)
plt.plot(time, control_history, label="Control Input", color='orange')
plt.title('Control Input vs Time (Nonlinear Control)')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (N)')
plt.legend()
 
plt.tight_layout()
plt.show()