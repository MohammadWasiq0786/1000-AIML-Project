"""
Project 666: Dexterous Manipulation
Description:
Dexterous manipulation involves performing tasks that require fine control over the robot’s gripper or hand, such as picking up small objects, adjusting their position, or performing complex actions. In this project, we will simulate dexterous manipulation using a robotic hand with multiple degrees of freedom (DOF). We will focus on simulating the robot's ability to adjust the grasp and manipulate objects with precision. The robot will use a basic inverse kinematics algorithm to adjust its hand's configuration for dexterous manipulation.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot Hand class with dexterous manipulation capabilities
class DexterousRobotHand:
    def __init__(self, fingers_length=1.0, hand_length=2.0):
        self.fingers_length = fingers_length  # Length of each finger
        self.hand_length = hand_length  # Length of the palm
        self.finger_angles = np.array([np.pi / 4, np.pi / 4, np.pi / 4])  # Initial angles of the three fingers
 
    def forward_kinematics(self):
        """
        Compute the positions of the fingertips based on the finger angles.
        :return: Coordinates of the fingertips
        """
        fingertips = []
        x, y = self.hand_length, 0  # Palm center (assume it's at the origin)
        
        # Calculate positions of each fingertip based on angles and length
        for angle in self.finger_angles:
            x_finger = x + self.fingers_length * np.cos(angle)
            y_finger = y + self.fingers_length * np.sin(angle)
            fingertips.append(np.array([x_finger, y_finger]))
 
        return np.array(fingertips)
 
    def inverse_kinematics(self, target_positions):
        """
        Compute the finger angles to reach the target positions.
        :param target_positions: Target positions for the fingertips
        :return: Finger angles to reach the target
        """
        # For simplicity, we'll assume that the robot's hand adjusts to target fingertip positions directly.
        angles = []
        for target in target_positions:
            # Calculate the angle to reach the target position (simple inverse kinematics)
            angle = np.arctan2(target[1], target[0])  # Angle in the plane
            angles.append(angle)
 
        self.finger_angles = np.array(angles)  # Update the finger angles
        return self.finger_angles
 
# 2. Define the Dexterous Manipulation Planner
class DexterousManipulationPlanner:
    def __init__(self, robot_hand):
        self.robot_hand = robot_hand
 
    def plan_manipulation(self, target_positions):
        """
        Plan the manipulation task to move the hand to the target positions.
        :param target_positions: Target positions for the fingertips
        :return: Joint angles for the robot hand to reach the target positions
        """
        # Compute inverse kinematics to find the finger angles
        angles = self.robot_hand.inverse_kinematics(target_positions)
        return angles
 
    def execute_plan(self, target_positions):
        """
        Simulate the execution of the manipulation task.
        :param target_positions: Target positions for the fingertips
        """
        # Plan the manipulation to move the hand to the target positions
        angles = self.plan_manipulation(target_positions)
 
        # Get the positions of the fingertips after the manipulation
        fingertips = self.robot_hand.forward_kinematics()
 
        # Visualize the hand movement
        self.plot_hand(fingertips)
 
    def plot_hand(self, fingertips):
        """
        Visualize the hand and its fingertips.
        :param fingertips: Positions of the fingertips
        """
        fig, ax = plt.subplots(figsize=(6, 6))
 
        # Plot the palm of the hand
        ax.plot([0, self.robot_hand.hand_length], [0, 0], label="Palm", color='blue', lw=4)
 
        # Plot each finger
        for finger in fingertips:
            ax.plot([self.robot_hand.hand_length, finger[0]], [0, finger[1]], label="Finger", color='green', lw=2)
 
        # Plot the fingertips
        for finger in fingertips:
            ax.scatter(finger[0], finger[1], color='red', s=100, label="Fingertip")
 
        ax.set_xlim(-self.robot_hand.hand_length - self.robot_hand.fingers_length, 
                    self.robot_hand.hand_length + self.robot_hand.fingers_length)
        ax.set_ylim(-self.robot_hand.fingers_length, 
                    self.robot_hand.fingers_length)
        
        ax.set_aspect('equal', 'box')
        ax.set_title("Dexterous Manipulation - Robotic Hand")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        plt.grid(True)
        plt.show()
 
# 3. Initialize the dexterous robot hand and manipulation planner
robot_hand = DexterousRobotHand(fingers_length=1.0, hand_length=2.0)
planner = DexterousManipulationPlanner(robot_hand)
 
# 4. Define target positions for the fingertips (for example, move the hand to a new position)
target_positions = np.array([[2, 2], [2.5, 2.5], [3, 3]])  # Target positions for the fingertips
 
# 5. Execute the task planning and visualization
planner.execute_plan(target_positions)