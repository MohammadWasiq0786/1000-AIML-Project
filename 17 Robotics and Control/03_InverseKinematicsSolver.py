"""
Project 643: Inverse Kinematics Solver
Description:
Inverse kinematics (IK) is the mathematical process of determining the joint angles of a robot's arm or manipulator to achieve a desired end effector position (e.g., reaching a specific point in space). Solving inverse kinematics is a fundamental problem in robotics, particularly for robotic arms and manipulators. In this project, we will implement an inverse kinematics solver that calculates the joint angles for a robotic arm to reach a target position using a numerical or analytical approach.
"""

import numpy as np
 
# 1. Define the robot arm with 2 joints
# The robot arm consists of two links with lengths l1 and l2
l1 = 1.0  # Length of first link
l2 = 1.0  # Length of second link
 
# 2. Inverse kinematics solver function for a 2D robotic arm
def inverse_kinematics(x, y):
    """
    Solves inverse kinematics for a 2-joint robotic arm in 2D.
    :param x: Desired x-coordinate of the end effector
    :param y: Desired y-coordinate of the end effector
    :return: joint angles theta1, theta2 (in radians)
    """
 
    # Calculate the distance to the target position (x, y)
    r = np.sqrt(x**2 + y**2)
 
    # Check if the target is reachable
    if r > (l1 + l2) or r < abs(l1 - l2):
        raise ValueError("Target is out of reach")
 
    # Use the law of cosines to calculate the angles
    cos_theta2 = (r**2 - l1**2 - l2**2) / (-2 * l1 * l2)
    theta2 = np.arccos(cos_theta2)
 
    # Calculate the angle of the first joint (theta1)
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
 
    return theta1, theta2
 
# 3. Test the inverse kinematics solver
target_x = 1.5  # Desired x-coordinate of the end effector
target_y = 1.0  # Desired y-coordinate of the end effector
 
# Solve the inverse kinematics to get joint angles
try:
    theta1, theta2 = inverse_kinematics(target_x, target_y)
    print(f"Joint 1 Angle (theta1): {np.degrees(theta1):.2f}°")
    print(f"Joint 2 Angle (theta2): {np.degrees(theta2):.2f}°")
except ValueError as e:
    print(e)