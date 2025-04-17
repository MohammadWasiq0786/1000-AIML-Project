"""
Project 665: Manipulation Planning
Description:
Manipulation planning is the process of planning the movements of a robot’s arm (or gripper) to manipulate objects. The robot needs to determine how to grasp, move, and place objects in the environment. In this project, we will implement a simple manipulation planning system for a robotic arm that can plan and execute basic movements to grasp and place an object. The system will use a kinematic model for planning and controlling the robot's arm.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot Arm class (2D manipulator for simplicity)
class RobotArm:
    def __init__(self, length1, length2, initial_angle1=0, initial_angle2=0):
        self.length1 = length1  # Length of the first arm segment
        self.length2 = length2  # Length of the second arm segment
        self.angle1 = initial_angle1  # Initial angle of the first arm segment
        self.angle2 = initial_angle2  # Initial angle of the second arm segment
 
    def forward_kinematics(self):
        """
        Compute the end effector position (x, y) based on the arm's joint angles.
        :return: (x, y) coordinates of the end effector
        """
        x = self.length1 * np.cos(self.angle1) + self.length2 * np.cos(self.angle1 + self.angle2)
        y = self.length1 * np.sin(self.angle1) + self.length2 * np.sin(self.angle1 + self.angle2)
        return np.array([x, y])
 
    def inverse_kinematics(self, target_position):
        """
        Compute the joint angles needed to reach a target position using inverse kinematics.
        :param target_position: The target (x, y) position for the end effector
        :return: joint angles (angle1, angle2)
        """
        x, y = target_position
        r = np.sqrt(x**2 + y**2)
        d = (r**2 - self.length1**2 - self.length2**2) / (2 * self.length1 * self.length2)
        angle2 = np.arccos(np.clip(d, -1.0, 1.0))  # Clamp value to avoid numerical errors
        angle1 = np.arctan2(y, x) - np.arctan2(self.length2 * np.sin(angle2), self.length1 + self.length2 * np.cos(angle2))
        return angle1, angle2
 
# 2. Define the Manipulation Planner
class ManipulationPlanner:
    def __init__(self, robot_arm):
        self.robot_arm = robot_arm
 
    def plan_manipulation(self, target_position):
        """
        Plan the manipulation task to move the robot arm to the target position.
        :param target_position: The target position (x, y) for the robot's end effector
        :return: Joint angles for the robot arm to reach the target position
        """
        # Compute inverse kinematics to find the joint angles
        angle1, angle2 = self.robot_arm.inverse_kinematics(target_position)
        return angle1, angle2
 
    def execute_plan(self, target_position):
        """
        Simulate the execution of the manipulation task.
        :param target_position: The target position (x, y) for the end effector
        """
        # Plan the manipulation to move the arm to the target position
        angle1, angle2 = self.plan_manipulation(target_position)
 
        # Set the joint angles of the arm
        self.robot_arm.angle1 = angle1
        self.robot_arm.angle2 = angle2
 
        # Get the end effector position after the movement
        end_effector_position = self.robot_arm.forward_kinematics()
 
        # Visualize the robot arm's movement
        self.plot_arm(end_effector_position)
 
    def plot_arm(self, end_effector_position):
        """
        Plot the robot arm and its end effector position.
        :param end_effector_position: The position of the robot's end effector
        """
        fig, ax = plt.subplots(figsize=(6, 6))
 
        # Robot arm base (origin)
        ax.plot([0, self.robot_arm.length1 * np.cos(self.robot_arm.angle1)], 
                [0, self.robot_arm.length1 * np.sin(self.robot_arm.angle1)], 
                label='Link 1', color='blue', lw=2)
 
        ax.plot([self.robot_arm.length1 * np.cos(self.robot_arm.angle1), 
                 end_effector_position[0]], 
                [self.robot_arm.length1 * np.sin(self.robot_arm.angle1), 
                 end_effector_position[1]], 
                label='Link 2', color='green', lw=2)
 
        # Plot the target position and the end effector
        ax.scatter(end_effector_position[0], end_effector_position[1], color='red', s=100, label='End Effector')
        ax.scatter(self.robot_arm.length1 * np.cos(self.robot_arm.angle1), 
                   self.robot_arm.length1 * np.sin(self.robot_arm.angle1), 
                   color='yellow', s=100, label='Joint 1')
 
        ax.scatter(0, 0, color='black', s=100, label='Base')
 
        ax.set_xlim(-self.robot_arm.length1 - self.robot_arm.length2, 
                    self.robot_arm.length1 + self.robot_arm.length2)
        ax.set_ylim(-self.robot_arm.length1 - self.robot_arm.length2, 
                    self.robot_arm.length1 + self.robot_arm.length2)
 
        ax.set_aspect('equal', 'box')
        ax.set_title("Robot Arm Manipulation")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        plt.grid(True)
        plt.show()
 
# 3. Initialize the robot arm and manipulation planner
robot_arm = RobotArm(length1=3, length2=2)
planner = ManipulationPlanner(robot_arm)
 
# 4. Define a target position and execute the plan
target_position = np.array([4, 2])  # Target position for the robot's end effector
planner.execute_plan(target_position)