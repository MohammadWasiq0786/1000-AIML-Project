"""
Project 740: Interpretable Reinforcement Learning
Description:
Interpretable reinforcement learning (RL) refers to methods that make the decision-making process of reinforcement learning models more transparent and understandable to humans. RL models typically involve complex decision-making, where agents learn to take actions to maximize cumulative rewards. However, these models are often considered black-box models, making it difficult to understand why certain actions are taken. In this project, we will implement a simple RL agent and introduce techniques to make its actions more interpretable, such as visualizing the value function, policy, and state-action pairs.

We will implement a Q-learning agent for a simple environment (e.g., FrozenLake from OpenAI's Gym). The goal is to visualize how the agent makes decisions based on its Q-values and policy, making it easier to understand its learning process.

Explanation:
Environment Creation: The create_environment() function initializes the FrozenLake environment from OpenAI's Gym. The goal is for the agent to navigate through the grid to reach the goal while avoiding the holes.

Q-table Initialization: The initialize_q_table() function initializes a Q-table that stores the Q-values for each state-action pair. Initially, all Q-values are set to 0.

Q-learning Algorithm:

The q_learning() function implements the Q-learning algorithm. It updates the Q-table based on the agent’s experience. The agent follows an epsilon-greedy strategy, where it either explores a random action or exploits the best-known action based on the current Q-table.

The Q-values are updated using the standard Q-learning update rule:

Q(s,a)=(1−α)Q(s,a)+α(R+γ⋅max⁡Q(s′,a′))Q(s, a) = (1 - \alpha) Q(s, a) + \alpha \left( R + \gamma \cdot \max Q(s', a') \right)Q(s,a)=(1−α)Q(s,a)+α(R+γ⋅maxQ(s′,a′))

where α\alphaα is the learning rate, γ\gammaγ is the discount factor, and RRR is the reward.

Q-table Visualization: The visualize_q_table() function visualizes the learned Q-table as a heatmap. Each state’s Q-values for all possible actions are shown, helping us understand which actions the model deems the best for each state.

Reward Visualization: We plot the total rewards over the episodes to observe how the agent’s performance improves over time.

This project provides an interpretable reinforcement learning model, where the Q-table and policy learned by the agent can be visualized and analyzed, making it easier to understand the model's decision-making process.
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
 
# 1. Create the FrozenLake environment
def create_environment():
    """
    Create the FrozenLake environment from OpenAI Gym.
    The environment is a 4x4 grid where the agent needs to reach the goal.
    """
    env = gym.make('FrozenLake-v1', is_slippery=False)
    return env
 
# 2. Initialize the Q-table
def initialize_q_table(state_space, action_space):
    """
    Initialize a Q-table with all zeros. 
    The Q-table stores the value of taking a certain action in a certain state.
    """
    q_table = np.zeros((state_space, action_space))
    return q_table
 
# 3. Q-learning algorithm to learn the Q-values
def q_learning(env, q_table, learning_rate=0.1, discount_factor=0.99, episodes=1000):
    """
    Perform Q-learning to update the Q-table based on the agent's experience.
    """
    total_rewards = []
 
    for episode in range(episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state
        done = False
        total_reward = 0
        
        while not done:
            # Choose an action using epsilon-greedy strategy
            if np.random.rand() < 0.1:
                action = env.action_space.sample()  # Explore: random action
            else:
                action = np.argmax(q_table[state])  # Exploit: choose best action based on Q-values
            
            # Take the chosen action and observe the new state and reward
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Update the Q-value for the state-action pair using the Q-learning update rule
            q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state]))
            state = next_state
        
        total_rewards.append(total_reward)
    
    return q_table, total_rewards
 
# 4. Visualize the Q-table (interpretability)
def visualize_q_table(q_table, action_space):
    """
    Visualize the learned Q-table. Each cell shows the Q-value for a specific state-action pair.
    """
    plt.imshow(q_table, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Q-Table Visualization')
    plt.xlabel('Action')
    plt.ylabel('State')
    plt.xticks(np.arange(action_space), ['Left', 'Down', 'Right', 'Up'])
    plt.show()
 
# 5. Example usage
env = create_environment()
 
# Initialize Q-table
q_table = initialize_q_table(state_space=env.observation_space.n, action_space=env.action_space.n)
 
# Train the agent with Q-learning
q_table, total_rewards = q_learning(env, q_table, episodes=1000)
 
# Visualize the learned Q-table
visualize_q_table(q_table, env.action_space.n)
 
# Plot the total rewards over episodes
plt.plot(total_rewards)
plt.title('Total Rewards over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()