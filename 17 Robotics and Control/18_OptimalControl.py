"""
Project 658: Optimal Control Implementation
Description:
Optimal control aims to find a control input that minimizes a cost function while satisfying the system's dynamics and constraints. It is widely used in applications where we want to optimize the performance of a system, such as energy efficiency, path planning, or economic cost. In this project, we will implement Optimal Control using the Linear Quadratic Regulator (LQR), a method used to solve optimal control problems with linear dynamics and quadratic cost functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
 
# 1. Define the system dynamics (A, B matrices for a linear system)
A = np.array([[0, 1], [0, -0.5]])  # State transition matrix (position, velocity)
B = np.array([[0], [1]])  # Control input matrix (force)
 
# 2. Define the LQR controller parameters
Q = np.array([[10, 0], [0, 1]])  # State cost matrix (penalizing position and velocity errors)
R = np.array([[1]])  # Control input cost matrix (penalizing control effort)
 
# 3. Define the LQR control law
def lqr(A, B, Q, R):
    """
    Solves the LQR optimal control problem.
    :param A: System dynamics matrix
    :param B: Control input matrix
    :param Q: State cost matrix
    :param R: Control input cost matrix
    :return: Optimal gain matrix K
    """
    P = np.linalg.solve(np.eye(A.shape[0]) * 1.0 + A.T @ P @ A + Q, R)  # Solve the Riccati equation
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A  # Compute the optimal feedback gain
    return K
 
# 4. Implement the LQR control for the system
class OptimalControlSystem:
    def __init__(self, A, B, Q, R, initial_state=[0, 0], target_state=[1, 0]):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.state = np.array(initial_state)  # Initial state (position, velocity)
        self.target_state = np.array(target_state)  # Target state (desired position, velocity)
        self.K = lqr(A, B, Q, R)  # Compute the optimal gain matrix K
        self.state_history = []  # Store history for plotting
 
    def apply_control(self, dt):
        """
        Apply the LQR control law and update the system state.
        :param dt: Time step for simulation.
        :return: Updated state and control input.
        """
        # Compute the control input using LQR
        control_input = -self.K @ (self.state - self.target_state)  # Control law: u = -K(x - x_target)
        
        # Update state based on system dynamics (dx = Ax + Bu)
        self.state = self.state + (self.A @ self.state + self.B @ control_input) * dt
        self.state_history.append(self.state[0])  # Store position for plotting
        return self.state, control_input
 
# 5. Initialize the optimal control system
oc_system = OptimalControlSystem(A, B, Q, R, initial_state=[0, 0], target_state=[1, 0])
 
# 6. Simulate the optimal control system over time
time = np.arange(0, 10, 0.1)  # Simulate for 10 seconds with a time step of 0.1s
position_history = []
control_history = []
 
for t in time:
    state, control_input = oc_system.apply_control(dt=0.1)  # 0.1s time step
    position_history.append(state[0])
    control_history.append(control_input[0])
 
# 7. Plot the results of the optimal control system
plt.figure(figsize=(10, 6))
 
# Plot position vs time
plt.subplot(2, 1, 1)
plt.plot(time, position_history, label="Position", color='blue')
plt.title('Position vs Time (Optimal Control with LQR)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
 
# Plot control input vs time
plt.subplot(2, 1, 2)
plt.plot(time, control_history, label="Control Input", color='green')
plt.title('Control Input vs Time (Optimal Control with LQR)')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (N)')
plt.legend()
 
plt.tight_layout()
plt.show()