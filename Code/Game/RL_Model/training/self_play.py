import numpy as np
import torch
from models.mcts import MCTS
from collections import deque
import random
from config import config
from game.othello import OthelloGame

class SelfPlayWorker:
    def __init__(self, network):
        self.network = network
        self.mcts = MCTS(network)
        self.game_buffer = deque(maxlen=config.buffer_size)
        
    def execute_episode(self, temperature=1.0):
        """Execute one episode of self-play"""
        game_states = []
        policies = []
        current_player = []
        
        game = OthelloGame()
        
        while not game.is_game_over():
            # Get MCTS policy
            policy = self.mcts.search(game)
            
            # Store state and policy
            game_states.append(game.get_state())
            policies.append(policy)
            current_player.append(game.current_player)
            
            # Select move based on policy and temperature
            valid_moves = game.get_valid_moves()
            policy_mask = np.zeros(config.action_size)
            for move in valid_moves:
                idx = move[0] * config.board_size + move[1]
                policy_mask[idx] = 1
            
            policy = policy * policy_mask
            if policy.sum() > 0:  # Ensure we have valid moves
                policy = policy / policy.sum()
                
                # Apply temperature
                if temperature == 0:
                    move_idx = policy.argmax()
                else:
                    policy = policy ** (1/temperature)
                    policy = policy / policy.sum()
                    move_idx = np.random.choice(len(policy), p=policy)
                
                move = (move_idx // config.board_size, 
                       move_idx % config.board_size)
                game.make_move(move)
            else:
                # No valid moves, must pass
                game.switch_player()
        
        # Get game outcome
        outcome = game.get_winner_value()
        
        # Create training examples
        for state, policy, player in zip(game_states, policies, current_player):
            value = outcome if player == game.current_player else -outcome
            self.game_buffer.append((state, policy, value))
        
        return game.get_winner()
    
    def get_batch(self, batch_size):
        """Sample batch of training examples"""
        if len(self.game_buffer) < batch_size:
            return None
            
        batch = random.sample(self.game_buffer, batch_size)
        states, policies, values = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(states)).to(config.device)
        policy_batch = torch.FloatTensor(np.array(policies)).to(config.device)
        value_batch = torch.FloatTensor(np.array(values)).to(config.device)
        
        return state_batch, policy_batch, value_batch