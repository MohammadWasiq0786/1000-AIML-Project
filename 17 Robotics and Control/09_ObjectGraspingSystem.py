"""
Project 649: Object Grasping System
Description:
An object grasping system allows a robot to identify and manipulate objects in its environment. The system typically involves perception (detecting the object and its position), planning (deciding where and how to grasp the object), and control (executing the grasp). In this project, we will simulate an object grasping system that uses a deep learning model for object detection and planning the best grasping points. The system will use the detected object’s position to move the robot’s gripper to grasp it.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
 
# 1. Simulate an object in the environment (for simplicity, we will define the object in a 2D space)
object_position = (3, 3)  # Example: Object position at (3, 3)
gripper_position = (0, 0)  # Start the gripper at (0, 0)
gripper_radius = 0.1  # Radius of the gripper (simulated)
 
# 2. Define the environment
class ObjectGraspingEnv:
    def __init__(self, object_position, gripper_position):
        self.object_position = object_position  # Object position
        self.gripper_position = gripper_position  # Gripper position
        self.gripper_radius = gripper_radius  # Gripper size
 
    def detect_object(self):
        # For simplicity, we assume the object is detected at the known position
        return self.object_position
 
    def calculate_grasp_position(self):
        # Calculate the best position to grasp the object (assumed to be the object's position)
        return self.object_position
 
    def move_gripper(self, grasp_position):
        # Move the gripper to the grasp position
        self.gripper_position = grasp_position
        return self.gripper_position
 
    def check_grasp_success(self):
        # Check if the gripper is close enough to the object to grasp it
        distance = np.linalg.norm(np.array(self.gripper_position) - np.array(self.object_position))
        if distance < self.gripper_radius:
            return True
        else:
            return False
 
# 3. Simulate the grasping process
env = ObjectGraspingEnv(object_position, gripper_position)
 
# Step 1: Detect object position
detected_object = env.detect_object()
print(f"Detected object position: {detected_object}")
 
# Step 2: Calculate the grasp position (for simplicity, it's the same as the object position)
grasp_position = env.calculate_grasp_position()
print(f"Calculated grasp position: {grasp_position}")
 
# Step 3: Move gripper to the grasp position
gripper_position = env.move_gripper(grasp_position)
print(f"Gripper moved to: {gripper_position}")
 
# Step 4: Check if the grasp is successful
if env.check_grasp_success():
    print("Grasp successful!")
else:
    print("Grasp failed. Retry.")
 
# 4. Visualize the environment and grasping process
fig, ax = plt.subplots(figsize=(6, 6))
plt.xlim(0, 5)
plt.ylim(0, 5)
 
# Plot the object and gripper
ax.scatter(*object_position, color='red', s=100, label="Object", zorder=5)
ax.scatter(*gripper_position, color='blue', s=100, label="Gripper", zorder=5)
 
# Plot gripper's radius of action
circle = plt.Circle(gripper_position, gripper_radius, color='blue', fill=False, linestyle='dashed', label="Gripper Radius")
ax.add_artist(circle)
 
# Plot labels
ax.legend()
ax.set_title("Object Grasping System")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
 
plt.grid(True)
plt.show()