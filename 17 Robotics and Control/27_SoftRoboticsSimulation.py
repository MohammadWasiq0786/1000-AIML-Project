"""
Project 667: Soft Robotics Simulation
Description:
Soft robotics involves robots built from flexible materials, designed to handle unstructured environments and delicate tasks, such as picking up fragile objects or squeezing through tight spaces. Soft robots are often designed to mimic biological organisms, offering enhanced dexterity and adaptability. In this project, we will simulate a simple soft robotic gripper using a soft actuated model to demonstrate basic manipulation and grasping tasks. The simulation will focus on controlling the gripper to handle objects of various shapes and sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Soft Robotic Gripper class
class SoftRoboticGripper:
    def __init__(self, finger_length=2.0, soft_material_stiffness=0.1):
        self.finger_length = finger_length  # Length of each finger
        self.soft_material_stiffness = soft_material_stiffness  # Flexibility of the gripper (soft material stiffness)
        self.finger_angles = np.array([np.pi / 4, np.pi / 4])  # Initial angles of the fingers
        self.target_position = None  # Target position to grasp an object
 
    def forward_kinematics(self):
        """
        Compute the positions of the fingertips based on the finger angles.
        :return: Coordinates of the fingertips
        """
        fingertips = []
        for angle in self.finger_angles:
            x_finger = self.finger_length * np.cos(angle)
            y_finger = self.finger_length * np.sin(angle)
            fingertips.append(np.array([x_finger, y_finger]))
        return np.array(fingertips)
 
    def set_target(self, position):
        """
        Set the target position for the gripper to grasp.
        :param position: Target position for the gripper
        """
        self.target_position = np.array(position)
 
    def adapt_grip(self):
        """
        Adapt the gripper's angles to suit the shape of the object to be grasped.
        :return: Adjusted finger angles to fit the object
        """
        if self.target_position is not None:
            distance = np.linalg.norm(self.target_position)
            # Adjust finger angles based on the distance to the object
            self.finger_angles = np.array([np.pi / 4 + self.soft_material_stiffness * distance,
                                           np.pi / 4 - self.soft_material_stiffness * distance])
        return self.finger_angles
 
    def simulate_grasp(self):
        """
        Simulate the grasping process by adapting the gripper's fingers.
        """
        if self.target_position is not None:
            # Adapt the gripper to the target position
            self.adapt_grip()
            fingertips = self.forward_kinematics()
            self.plot_gripper(fingertips)
 
    def plot_gripper(self, fingertips):
        """
        Plot the soft robotic gripper and the object being grasped.
        :param fingertips: Positions of the fingertips
        """
        fig, ax = plt.subplots(figsize=(6, 6))
 
        # Plot the gripper's fingers
        ax.plot([0, fingertips[0][0]], [0, fingertips[0][1]], label="Finger 1", color='blue', lw=2)
        ax.plot([0, fingertips[1][0]], [0, fingertips[1][1]], label="Finger 2", color='green', lw=2)
 
        # Plot the object to be grasped (target position)
        ax.scatter(self.target_position[0], self.target_position[1], color='red', s=100, label="Object")
 
        ax.set_xlim(-self.finger_length - 1, self.finger_length + 1)
        ax.set_ylim(-self.finger_length - 1, self.finger_length + 1)
 
        ax.set_aspect('equal', 'box')
        ax.set_title("Soft Robotic Gripper - Grasping Simulation")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        plt.grid(True)
        plt.show()
 
# 2. Initialize the soft robotic gripper and set the target position
soft_gripper = SoftRoboticGripper(finger_length=2.0, soft_material_stiffness=0.1)
 
# Set the target position (position of the object to be grasped)
target_position = np.array([2.5, 3.0])  # Example target position for grasping
soft_gripper.set_target(target_position)
 
# 3. Simulate the grasping process
soft_gripper.simulate_grasp()