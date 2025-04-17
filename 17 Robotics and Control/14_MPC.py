"""
Project 654: Model Predictive Control
Description:
Model Predictive Control (MPC) is an advanced control strategy that uses a model of the system to predict its future behavior and optimize the control inputs over a prediction horizon. In MPC, the control inputs are computed by solving an optimization problem at each time step to minimize a cost function, typically involving state and control input errors. This technique is widely used in robotics, autonomous vehicles, and process control. In this project, we will implement a simple Model Predictive Controller for a robotic system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
 
# 1. Define the system model (simple linear system: x(t+1) = Ax(t) + Bu(t))
A = np.array([[1, 1], [0, 1]])  # System dynamics (state transition matrix)
B = np.array([[0.5], [1]])  # Control input matrix
 
# 2. Define the Model Predictive Control class
class MPC:
    def __init__(self, N, Q, R, x0, u_max=1.0):
        """
        Initialize the MPC controller.
        :param N: Prediction horizon
        :param Q: State error cost matrix
        :param R: Control input cost matrix
        :param x0: Initial state
        :param u_max: Maximum control input
        """
        self.N = N  # Prediction horizon
        self.Q = Q  # State error cost matrix
        self.R = R  # Control input cost matrix
        self.x0 = x0  # Initial state
        self.u_max = u_max  # Maximum control input
        self.x = x0  # Current state
        self.u = np.zeros((N, 1))  # Control inputs to optimize
 
    def objective_function(self, u):
        """
        Objective function for MPC optimization.
        :param u: Control inputs
        :return: The cost function value
        """
        cost = 0
        x = self.x0  # Reset state at the beginning of each optimization
 
        # Calculate the cost over the prediction horizon
        for k in range(self.N):
            x = A @ x + B * u[k]  # Update state based on the model
            cost += x.T @ self.Q @ x + u[k].T @ self.R @ u[k]  # Add state and control input costs
 
        return cost
 
    def solve(self):
        """
        Solve the optimization problem to compute the optimal control input.
        """
        # Optimize control inputs (u) over the prediction horizon
        result = minimize(self.objective_function, np.zeros(self.N), bounds=[(-self.u_max, self.u_max)] * self.N)
 
        if result.success:
            self.u = result.x  # Update the control inputs
        else:
            print("Optimization failed.")
        
        # Apply the first control input and update the state
        u_opt = self.u[0]
        self.x = A @ self.x + B * u_opt
        return u_opt, self.x
 
# 3. Initialize the MPC controller
N = 10  # Prediction horizon (steps)
Q = np.diag([1, 1])  # State error cost matrix (penalize deviations in position and velocity)
R = np.array([[0.1]])  # Control input cost matrix (penalize large control inputs)
x0 = np.array([0, 0])  # Initial state (position = 0, velocity = 0)
 
mpc = MPC(N, Q, R, x0)
 
# 4. Simulate the robot with MPC control
time = np.arange(0, 30, 1)  # Simulate for 30 time steps
position_history = []
velocity_history = []
control_history = []
 
for t in time:
    # Solve the MPC optimization to get the control input
    u_opt, state = mpc.solve()
 
    # Store the results for plotting
    position_history.append(state[0])
    velocity_history.append(state[1])
    control_history.append(u_opt)
 
# 5. Plot the results of the MPC controller
plt.figure(figsize=(10, 6))
 
# Plot position vs time
plt.subplot(3, 1, 1)
plt.plot(time, position_history, label="Position", color='blue')
plt.title('Position vs Time (MPC Control)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
 
# Plot velocity vs time
plt.subplot(3, 1, 2)
plt.plot(time, velocity_history, label="Velocity", color='green')
plt.title('Velocity vs Time (MPC Control)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
 
# Plot control input vs time
plt.subplot(3, 1, 3)
plt.plot(time, control_history, label="Control Input", color='orange')
plt.title('Control Input vs Time (MPC Control)')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.legend()
 
plt.tight_layout()
plt.show()