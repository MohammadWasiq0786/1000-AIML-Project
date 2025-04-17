"""
Project 675: Robot Learning System
Description:
A robot learning system enables robots to improve their performance over time by learning from experience or demonstration. In this project, we will implement a simple learning system for a robot to adapt to an environment. We will use reinforcement learning (RL) to teach the robot how to navigate a grid environment and reach a target location while avoiding obstacles. The robot will use trial-and-error learning to improve its ability to navigate the environment.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define the Robot Learning System class using Q-learning
class RobotLearningSystem:
    def __init__(self, grid_size=(5, 5), start_position=(0, 0), goal_position=(4, 4), num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.grid_size = grid_size  # Size of the grid environment
        self.start_position = np.array(start_position)  # Initial position of the robot
        self.goal_position = np.array(goal_position)  # Goal position
        self.position = self.start_position  # Current position of the robot
        self.num_episodes = num_episodes  # Number of training episodes
        self.learning_rate = learning_rate  # Learning rate for Q-learning
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration-exploitation tradeoff parameter
        
        # Initialize Q-table (action-value table)
        self.q_table = np.zeros((grid_size[0], grid_size[1], 4))  # 4 actions: up, down, left, right
 
    def get_possible_actions(self):
        """
        Return a list of possible actions (up, down, left, right) for the robot at the current position.
        """
        actions = []
        x, y = self.position
        if x > 0: actions.append(0)  # Up
        if x < self.grid_size[0] - 1: actions.append(1)  # Down
        if y > 0: actions.append(2)  # Left
        if y < self.grid_size[1] - 1: actions.append(3)  # Right
        return actions
 
    def take_action(self, action):
        """
        Perform the action and update the robot's position.
        :param action: The action to perform (0 = up, 1 = down, 2 = left, 3 = right)
        """
        x, y = self.position
        if action == 0:  # Up
            self.position = np.array([x - 1, y])
        elif action == 1:  # Down
            self.position = np.array([x + 1, y])
        elif action == 2:  # Left
            self.position = np.array([x, y - 1])
        elif action == 3:  # Right
            self.position = np.array([x, y + 1])
 
    def get_reward(self):
        """
        Calculate the reward based on the robot's current position.
        """
        if np.array_equal(self.position, self.goal_position):
            return 10  # Reward for reaching the goal
        else:
            return -1  # Penalty for each step
 
    def epsilon_greedy(self):
        """
        Select an action using the epsilon-greedy policy.
        :return: The chosen action
        """
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action
            possible_actions = self.get_possible_actions()
            return np.random.choice(possible_actions)
        else:
            # Exploitation: Choose the action with the highest Q-value
            x, y = self.position
            return np.argmax(self.q_table[x, y])  # Return the action with the max Q-value
 
    def update_q_table(self, action, reward, next_position):
        """
        Update the Q-table using the Q-learning update rule.
        :param action: The action that was taken
        :param reward: The reward received after taking the action
        :param next_position: The new position after taking the action
        """
        x, y = self.position
        next_x, next_y = next_position
        best_next_action = np.argmax(self.q_table[next_x, next_y])  # Best action in the next state
        # Q-learning update rule
        self.q_table[x, y, action] = self.q_table[x, y, action] + self.learning_rate * (reward + self.discount_factor * self.q_table[next_x, next_y, best_next_action] - self.q_table[x, y, action])
 
    def train(self):
        """
        Train the robot using Q-learning.
        """
        for episode in range(self.num_episodes):
            self.position = self.start_position  # Reset the robot's position at the start of each episode
            total_reward = 0
 
            while not np.array_equal(self.position, self.goal_position):  # Until the robot reaches the goal
                action = self.epsilon_greedy()  # Select an action
                old_position = self.position.copy()
                self.take_action(action)  # Take the action
                reward = self.get_reward()  # Get the reward for the action
                total_reward += reward
                self.update_q_table(action, reward, self.position)  # Update the Q-table
 
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.num_episodes} - Total Reward: {total_reward}")
 
    def plot(self):
        """
        Visualize the robot's learning progress on the grid.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(np.zeros(self.grid_size), cmap='gray', origin='upper')
        plt.scatter(self.goal_position[1], self.goal_position[0], color='red', s=100, label="Goal")
        plt.scatter(self.start_position[1], self.start_position[0], color='blue', s=100, label="Start")
 
        # Plot the learned path (for simplicity, we show the robot's final position)
        plt.scatter(self.position[1], self.position[0], color='green', s=100, label="Final Position")
        plt.legend()
        plt.title("Robot Learning System - Q-Learning")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()
 
# 2. Initialize the robot learning system and train it
robot_learning = RobotLearningSystem(grid_size=(5, 5), start_position=(0, 0), goal_position=(4, 4))
robot_learning.train()  # Train the robot using Q-learning
 
# 3. Plot the robot's final position after training
robot_learning.plot()