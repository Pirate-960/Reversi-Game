import os
from dataclasses import dataclass
import torch

@dataclass
class Config:
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = max(1, os.cpu_count() - 1)
    
    # Game
    board_size: int = 8
    action_size: int = board_size * board_size
    
    # Network
    num_channels: int = 3
    num_res_blocks: int = 19
    hidden_channels: int = 256
    
    # MCTS
    num_simulations: int = 800  # Reduced for initial testing
    c_puct: float = 4.0
    
    # Training
    num_episodes: int = 1000
    batch_size: int = 512
    buffer_size: int = 100000
    
    # Learning rates and weights
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_weight: float = 1.5
    value_weight: float = 1.0
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

config = Config()