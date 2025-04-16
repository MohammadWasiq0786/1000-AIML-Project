"""
Project 252. Monte Carlo tree search
Description:
Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for decision-making in discrete action spaces — especially in games like Go, Chess, and Tic-Tac-Toe. MCTS balances exploration and exploitation by simulating multiple game outcomes (rollouts), gradually building a search tree, and choosing actions with the highest win rates or values.

In this project, we’ll implement MCTS for Tic-Tac-Toe, a simple but effective domain to visualize the tree-based planning logic.

About:
✅ What It Does:
Simulates games from each node to estimate action value.

Balances exploration vs. exploitation using UCT (Upper Confidence Bound).

Expands search tree dynamically during gameplay.

Powers intelligent decision-making in games and planning systems.
"""


import random
import math
import copy
 
# Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = [" "] * 9
        self.current_player = "X"
 
    def available_actions(self):
        return [i for i, val in enumerate(self.board) if val == " "]
 
    def move(self, action):
        if self.board[action] != " ":
            return False
        self.board[action] = self.current_player
        self.current_player = "O" if self.current_player == "X" else "X"
        return True
 
    def winner(self):
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for i,j,k in lines:
            if self.board[i] == self.board[j] == self.board[k] != " ":
                return self.board[i]
        return None
 
    def terminal(self):
        return self.winner() is not None or " " not in self.board
 
    def clone(self):
        clone = TicTacToe()
        clone.board = self.board[:]
        clone.current_player = self.current_player
        return clone
 
    def print_board(self):
        print("\n".join(["|".join(self.board[i:i+3]) for i in range(0,9,3)]))
 
# MCTS Node
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
 
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.available_actions())
 
    def best_child(self, c_param=1.4):
        choices = []
        for child in self.children:
            uct = (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-4))
            choices.append((uct, child))
        return max(choices, key=lambda x: x[0])[1]
 
    def expand(self):
        tried_actions = [child.action for child in self.children]
        for action in self.state.available_actions():
            if action not in tried_actions:
                next_state = self.state.clone()
                next_state.move(action)
                child_node = MCTSNode(next_state, parent=self, action=action)
                self.children.append(child_node)
                return child_node
        return None
 
    def simulate(self):
        sim_state = self.state.clone()
        while not sim_state.terminal():
            action = random.choice(sim_state.available_actions())
            sim_state.move(action)
        winner = sim_state.winner()
        return winner
 
    def backpropagate(self, result, player):
        self.visits += 1
        if result == player:
            self.wins += 1
        elif result is None:
            self.wins += 0.5  # draw
        if self.parent:
            self.parent.backpropagate(result, player)
 
# MCTS Algorithm
def mcts_search(initial_state, iter_limit=1000):
    root = MCTSNode(initial_state)
 
    for _ in range(iter_limit):
        node = root
 
        # Selection
        while node.is_fully_expanded() and not node.state.terminal():
            node = node.best_child()
 
        # Expansion
        if not node.state.terminal():
            node = node.expand()
 
        # Simulation
        result = node.simulate()
 
        # Backpropagation
        node.backpropagate(result, initial_state.current_player)
 
    return max(root.children, key=lambda c: c.visits).action
 
# Play a game using MCTS vs Random
game = TicTacToe()
while not game.terminal():
    game.print_board()
    print(f"\nCurrent Player: {game.current_player}")
 
    if game.current_player == "X":
        move = mcts_search(game, iter_limit=500)
    else:
        move = random.choice(game.available_actions())
 
    print(f"Selected Move: {move}")
    game.move(move)
 
game.print_board()
winner = game.winner()
print(f"\n🏁 Game Over! Winner: {winner if winner else 'Draw'}")