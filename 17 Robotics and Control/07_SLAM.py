"""
Project 647: Simultaneous Localization and Mapping (SLAM)
Description:
Simultaneous Localization and Mapping (SLAM) is a technique used by robots to build a map of an unknown environment while simultaneously keeping track of its own position within that environment. This is essential for tasks such as autonomous navigation. In this project, we will implement a simple SLAM algorithm that estimates the robot's trajectory and the environment's map. We will use Extended Kalman Filter (EKF) for localization and mapping, which is a widely used method for SLAM in robotics.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the robot motion model
def motion_model(state, control_input, dt):
    """
    Motion model: Predict the new state of the robot.
    :param state: Current state (x, y, theta)
    :param control_input: Control input (linear velocity, angular velocity)
    :param dt: Time step
    :return: New state after applying control
    """
    x, y, theta = state
    v, w = control_input
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dtheta = w * dt
    return np.array([x + dx, y + dy, theta + dtheta])
 
# 2. Define the process noise model
def process_noise(state, control_input, dt):
    """
    Process noise model: Add noise to the control inputs.
    :param state: Current state (x, y, theta)
    :param control_input: Control input (linear velocity, angular velocity)
    :param dt: Time step
    :return: Process noise covariance matrix
    """
    v, w = control_input
    Q = np.diag([0.1*v**2, 0.1*w**2])  # Simple noise model for velocity and angular velocity
    return Q
 
# 3. Define the measurement model (landmark observations)
def measurement_model(state, landmark):
    """
    Measurement model: Predict the measurement (distance and angle to a landmark).
    :param state: Current state (x, y, theta)
    :param landmark: Landmark position (lx, ly)
    :return: Measurement (distance, angle)
    """
    x, y, theta = state
    lx, ly = landmark
    dx = lx - x
    dy = ly - y
    r = np.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx) - theta
    return np.array([r, alpha])
 
# 4. Define the observation noise model
def observation_noise():
    """
    Observation noise model: Add noise to the measurements.
    :return: Observation noise covariance matrix
    """
    R = np.diag([0.1, 0.05])  # Noise in distance and angle measurements
    return R
 
# 5. Extended Kalman Filter for SLAM
class EKF_SLAM:
    def __init__(self, initial_state, landmarks, dt=1.0):
        self.state = initial_state  # Initial state (x, y, theta)
        self.covariance = np.eye(3) * 0.1  # Initial covariance matrix
        self.landmarks = landmarks  # List of landmarks
        self.dt = dt  # Time step
        self.history = []  # To store the robot's trajectory
 
    def predict(self, control_input):
        """
        Predict the next state of the robot using the motion model.
        :param control_input: Control input (v, w)
        """
        self.state = motion_model(self.state, control_input, self.dt)  # Update state
        Q = process_noise(self.state, control_input, self.dt)  # Process noise
        self.covariance = self.covariance + Q  # Update covariance
 
    def update(self, measurement, landmark_idx):
        """
        Update the state based on the measurement and landmark observation.
        :param measurement: Actual measurement (distance, angle)
        :param landmark_idx: The index of the observed landmark
        """
        # Calculate the expected measurement
        predicted_measurement = measurement_model(self.state, self.landmarks[landmark_idx])
        H = np.array([[-(self.state[0] - self.landmarks[landmark_idx][0]) / predicted_measurement[0], 
                       -(self.state[1] - self.landmarks[landmark_idx][1]) / predicted_measurement[0], 0],
                      [self.state[1] - self.landmarks[landmark_idx][1], 
                       -(self.state[0] - self.landmarks[landmark_idx][0]), -predicted_measurement[0]]]) / predicted_measurement[0]
        R = observation_noise()  # Observation noise
 
        # Kalman Gain
        S = np.dot(np.dot(H, self.covariance), H.T) + R  # Innovation covariance
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))  # Kalman gain
 
        # Update state and covariance
        y = measurement - predicted_measurement  # Innovation (measurement residual)
        self.state = self.state + np.dot(K, y)  # Updated state
        self.covariance = self.covariance - np.dot(np.dot(K, H), self.covariance)  # Updated covariance
 
        self.history.append(self.state[:2])  # Store robot's position for plotting
 
# 6. Initialize EKF_SLAM with a robot, landmarks, and initial state
initial_state = np.array([0, 0, 0])  # Start position (x, y, theta)
landmarks = [(3, 3), (6, 1), (8, 8)]  # Landmarks positions
ekf_slam = EKF_SLAM(initial_state, landmarks)
 
# 7. Simulate the robot's motion and SLAM process
num_steps = 50
for step in range(num_steps):
    # Simulated control inputs (linear velocity, angular velocity)
    control_input = np.array([0.2, 0.1])  # Move forward and rotate slightly
 
    # Predict the next state
    ekf_slam.predict(control_input)
 
    # Simulate measurement of a random landmark
    landmark_idx = np.random.choice(len(landmarks))  # Randomly choose a landmark
    measurement = measurement_model(ekf_slam.state, landmarks[landmark_idx]) + np.random.randn(2) * 0.1  # Add noise to the measurement
 
    # Update the state with the measurement
    ekf_slam.update(measurement, landmark_idx)
 
# 8. Plot the results of the SLAM process
trajectory = np.array(ekf_slam.history)
landmarks = np.array(landmarks)
 
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Robot Trajectory", color="blue")
plt.scatter(landmarks[:, 0], landmarks[:, 1], color="red", marker="x", label="Landmarks")
plt.title("Simultaneous Localization and Mapping (SLAM) with EKF")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.show()