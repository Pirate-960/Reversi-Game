import numpy as np
import torch
from config import config
import math
from copy import deepcopy

class MCTSNode:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = {}  # Dictionary of {action: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0
        self.is_terminal = game_state.is_game_over()
        
    def expand(self, policy):
        """Expand node with predicted policy"""
        valid_moves = self.game_state.get_valid_moves()
        for move in valid_moves:
            if move not in self.children:
                new_state = deepcopy(self.game_state)
                new_state.make_move(move)
                self.children[move] = MCTSNode(new_state, parent=self)
                # Get index in flattened policy
                idx = move[0] * config.board_size + move[1]
                self.children[move].prior = policy[idx]

    def select_child(self):
        """Select child using PUCT algorithm"""
        best_score = float('-inf')
        best_action = None
        best_child = None

        # Calculate UCB for all children
        for action, child in self.children.items():
            ucb_score = self._ucb_score(child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(self, child):
        """Calculate UCB score for a child"""
        prior_score = (config.c_puct * child.prior * 
                      math.sqrt(self.visit_count) / (1 + child.visit_count))
        value_score = 0
        if child.visit_count > 0:
            value_score = -child.value_sum / child.visit_count
        return value_score + prior_score

    def backpropagate(self, value):
        """Update node statistics"""
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, network):
        self.network = network
        
    def search(self, game_state):
        root = MCTSNode(game_state)
        
        # Initial policy prediction
        state_tensor = self._prepare_state(game_state)
        with torch.no_grad():
            policy, _ = self.network(state_tensor)
            policy = policy.cpu().numpy().squeeze()
        
        root.expand(policy)
        
        # Run simulations
        for _ in range(config.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.children and not node.is_terminal:
                action, node = node.select_child()
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.is_terminal:
                state_tensor = self._prepare_state(node.game_state)
                with torch.no_grad():
                    policy, value = self.network(state_tensor)
                    policy = policy.cpu().numpy().squeeze()
                    value = value.item()
                node.expand(policy)
            else:
                # Game is over, use true game outcome
                value = node.game_state.get_winner_value()
            
            # Backpropagation
            for node in reversed(search_path):
                node.backpropagate(value)
                value = -value
        
        # Calculate improved policy
        visit_counts = np.zeros(config.action_size)
        for action, child in root.children.items():
            idx = action[0] * config.board_size + action[1]
            visit_counts[idx] = child.visit_count
        
        # Normalize to get probabilities
        policy = visit_counts / visit_counts.sum()
        
        return policy

    def _prepare_state(self, game_state):
        """Convert game state to network input format"""
        state = np.zeros((config.num_channels, 
                         config.board_size, 
                         config.board_size))
        
        # Current player pieces
        state[0] = (game_state.board == game_state.current_player)
        # Opponent pieces
        state[1] = (game_state.board == -game_state.current_player)
        # Empty spaces
        state[2] = (game_state.board == 0)
        
        return torch.FloatTensor(state).unsqueeze(0).to(config.device)