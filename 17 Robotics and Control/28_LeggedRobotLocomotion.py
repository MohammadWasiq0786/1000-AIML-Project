"""
Project 668: Legged Robot Locomotion
Description:
Legged robot locomotion involves creating algorithms that allow robots with legs to walk or move efficiently. Unlike wheeled robots, legged robots face the challenge of maintaining stability and adapting to different terrain. In this project, we will simulate a quadruped robot (four-legged robot) locomotion system. The robot will use basic control strategies to move forward, and we will implement a simple gait generation method to control its legs.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Quadruped Robot class
class QuadrupedRobot:
    def __init__(self, leg_length=1.0, stride_length=2.0):
        self.leg_length = leg_length  # Length of each leg
        self.stride_length = stride_length  # Length of the stride (distance per step)
        self.position = np.array([0, 0])  # Initial position of the robot (x, y)
        self.orientation = 0  # Initial orientation (in radians)
        self.gait_phase = 0  # Phase of the gait (for controlling leg movements)
 
    def move_forward(self):
        """
        Simulate forward movement by updating the position.
        """
        # Update position based on stride length and orientation
        self.position += np.array([self.stride_length * np.cos(self.orientation),
                                   self.stride_length * np.sin(self.orientation)])
 
    def update_orientation(self):
        """
        Update the orientation of the robot based on the gait phase.
        """
        # Simple gait control: change orientation at each phase (for simplicity)
        self.orientation += np.pi / 4  # Adjust orientation (robot turns after each step)
        if self.orientation > 2 * np.pi:
            self.orientation -= 2 * np.pi  # Keep orientation within [0, 2*pi]
 
    def generate_gait(self):
        """
        Simulate a gait generation for the quadruped robot.
        """
        # Adjust gait phase and update robot movement
        self.gait_phase += 1
        if self.gait_phase % 2 == 0:  # Alternate leg movements (simple gait pattern)
            self.move_forward()
        self.update_orientation()
 
    def plot(self):
        """
        Visualize the robot's movement on a 2D plane.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(self.position[0], self.position[1], color='blue', s=100, label="Robot Position")
        plt.quiver(self.position[0], self.position[1], np.cos(self.orientation), np.sin(self.orientation), 
                   angles='xy', scale_units='xy', scale=0.1, color='red', label="Robot Orientation")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title("Legged Robot Locomotion")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.show()
 
# 2. Initialize the quadruped robot and simulate its movement
robot = QuadrupedRobot(leg_length=1.0, stride_length=0.5)
 
# 3. Simulate the robot moving forward for 20 steps
for step in range(20):
    robot.generate_gait()
    robot.plot()