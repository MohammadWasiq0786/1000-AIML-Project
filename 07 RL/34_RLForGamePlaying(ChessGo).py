"""
Project 274. RL for game playing (Chess/Go)
Description:
Game-playing is one of the most iconic applications of RL. From AlphaGo to AlphaZero, agents have mastered board games by learning self-play strategies, using techniques like Monte Carlo Tree Search (MCTS) and Deep Q-learning. In this project, we’ll simplify things and build an RL agent that learns to play Tic-Tac-Toe through self-play using Q-learning.

Although not as complex as Chess or Go, the same RL principles apply — it's a great base for expanding into full-fledged game AI.

About:
✅ What It Does:
Trains two Q-learning agents via self-play.

Updates Q-values based on game outcomes (win, lose, draw).

Learns optimal strategies for Tic-Tac-Toe — generalizable to Chess/Go.

Foundation for AlphaZero-style self-play learning loops.
"""


import numpy as np
import random
import matplotlib.pyplot as plt
 
# Simple Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.board = [' ']*9
        self.current_player = 'X'
        return self.get_state()
 
    def get_state(self):
        return ''.join(self.board)
 
    def available_actions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']
 
    def step(self, action):
        self.board[action] = self.current_player
        reward, done = self.check_win_draw()
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return self.get_state(), reward, done
 
    def check_win_draw(self):
        wins = [(0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)]
        for i, j, k in wins:
            if self.board[i] == self.board[j] == self.board[k] != ' ':
                return (1 if self.board[i] == 'X' else -1), True
        if ' ' not in self.board:
            return 0, True  # draw
        return 0, False
 
# Q-learning agent for Tic-Tac-Toe
class QAgent:
    def __init__(self, name, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.name = name
        self.q = {}  # state-action value table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
 
    def get_qs(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(9)
        return self.q[state]
 
    def choose_action(self, state, valid_actions):
        q_values = self.get_qs(state)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        mask = np.full(9, -np.inf)
        mask[valid_actions] = q_values[valid_actions]
        return int(np.argmax(mask))
 
    def update(self, state, action, reward, next_state, next_valid):
        max_q_next = 0 if next_state is None else np.max(self.get_qs(next_state)[next_valid])
        td_target = reward + self.gamma * max_q_next
        self.get_qs(state)[action] += self.alpha * (td_target - self.get_qs(state)[action])
 
# Self-play training
episodes = 10000
player_X = QAgent("X")
player_O = QAgent("O")
reward_log = []
 
for ep in range(episodes):
    game = TicTacToe()
    state = game.get_state()
    done = False
    history = []
 
    while not done:
        current_agent = player_X if game.current_player == 'X' else player_O
        valid = game.available_actions()
        action = current_agent.choose_action(state, valid)
        next_state, reward, done = game.step(action)
        next_valid = game.available_actions() if not done else []
 
        # Record transition
        history.append((current_agent, state, action, reward, next_state, next_valid))
        state = next_state
 
    # Final reward assignment
    for agent, s, a, r, ns, nv in reversed(history):
        agent.update(s, a, r, ns if not done else None, nv)
        r = -r  # opponent gets negative reward
 
    reward_log.append(history[0][3])  # reward from X’s perspective
 
    if (ep + 1) % 1000 == 0:
        win_rate = sum([1 for r in reward_log[-1000:] if r == 1]) / 10
        print(f"Episode {ep+1}, X win rate: {win_rate:.1f}%")
 
# Plot learning trend
win_ratio = [sum([1 for r in reward_log[i:i+100] if r == 1])/100 for i in range(0, len(reward_log)-100)]
plt.plot(win_ratio)
plt.title("Tic-Tac-Toe: RL Agent (X) Self-Play Win Rate")
plt.xlabel("Training Batches (x100)")
plt.ylabel("Win Rate (%)")
plt.grid(True)
plt.show()