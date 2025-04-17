"""
Project 656: Robust Control Systems
Description:
Robust control focuses on designing control systems that can handle uncertainties in the system dynamics or disturbances in the environment without sacrificing performance. In this project, we will implement a robust control system for a robotic system that ensures the robot remains stable and performs optimally even with modeling errors or external disturbances. We will use a simple H∞ control method, which aims to minimize the worst-case performance of the system, making it robust to uncertainties.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
 
# 1. Define the system dynamics (A, B matrices for a simple mass-spring-damper system)
A = np.array([[0, 1], [-2, -0.5]])  # System dynamics matrix
B = np.array([[0], [1]])  # Control input matrix
C = np.array([1, 0])  # Output matrix (position)
 
# 2. Define the H∞ control design parameters
Q = np.eye(2)  # State cost matrix (penalizing the state)
R = np.array([[1]])  # Control input cost matrix (penalizing control efforts)
P = np.eye(2)  # Placeholder for the solution to the Riccati equation
Gamma = 1.0  # Weighting factor for robustness
 
# 3. Define the H∞ control algorithm
def h_infinity_control(A, B, C, Q, R, P, Gamma):
    """
    A simple implementation of H∞ control design.
    :param A: System dynamics matrix
    :param B: Control input matrix
    :param C: Output matrix
    :param Q: State cost matrix
    :param R: Control input cost matrix
    :param P: Solution to the Riccati equation (initial guess)
    :param Gamma: Robustness weighting factor
    :return: Gain matrix K for the controller
    """
    # Solve the Riccati equation (simple approximation here for demonstration)
    P_new = inv(A.T @ P @ A + Q + Gamma * B @ inv(R) @ B.T)
    K = inv(R + B.T @ P_new @ B) @ B.T @ P_new @ A  # Calculate the feedback gain K
 
    return K, P_new
 
# 4. Implement the closed-loop system with H∞ control
class RobustControlSystem:
    def __init__(self, A, B, C, K, desired_position=1.0):
        self.A = A
        self.B = B
        self.C = C
        self.K = K
        self.position = 0.0  # Initial position
        self.velocity = 0.0  # Initial velocity
        self.desired_position = desired_position  # Desired position
        self.state = np.array([self.position, self.velocity])
 
    def apply_control(self, dt):
        """
        Apply the H∞ control law to the system.
        :param dt: Time step for simulation
        """
        # Compute the control input using the state feedback
        control_input = -self.K @ self.state
        # Simulate the dynamics of the system
        self.state = self.state + np.array([self.velocity, -2 * self.position - 0.5 * self.velocity + control_input]) * dt
        self.position, self.velocity = self.state
 
# 5. Initialize the H∞ controller
K, P = h_infinity_control(A, B, C, Q, R, P, Gamma)
 
# 6. Simulate the robot with robust control over time
time = np.arange(0, 10, 0.1)  # Simulate for 10 seconds with a time step of 0.1s
position_history = []
velocity_history = []
 
# Initialize the robust control system
robust_control = RobustControlSystem(A, B, C, K)
 
for t in time:
    # Apply the H∞ control and update the system state
    robust_control.apply_control(dt=0.1)  # 0.1s time step
    position_history.append(robust_control.position)
    velocity_history.append(robust_control.velocity)
 
# 7. Plot the results of the robust control system
plt.figure(figsize=(10, 6))
 
# Plot position vs time
plt.subplot(2, 1, 1)
plt.plot(time, position_history, label="Position", color='blue')
plt.title('Position vs Time (Robust Control)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
 
# Plot velocity vs time
plt.subplot(2, 1, 2)
plt.plot(time, velocity_history, label="Velocity", color='green')
plt.title('Velocity vs Time (Robust Control)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
 
plt.tight_layout()
plt.show()