import os
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # System
    num_workers: int = os.cpu_count()
    use_gpu: bool = True
    mixed_precision: bool = True
    
    # Network
    num_channels: int = 3
    num_res_blocks: int = 19
    channels: int = 256
    
    # Game
    board_size: int = 8
    action_size: int = board_size * board_size
    
    # MCTS
    num_simulations: int = 1600
    c_puct: float = 4.0
    
    # Training
    num_episodes: int = 5000
    batch_size: int = 2048
    buffer_size: int = 500000
    
    # Temperature
    initial_temp: float = 1.0
    min_temp: float = 0.1
    temp_decay: float = 0.97
    
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_weight: float = 1.5
    value_weight: float = 1.0
    
    # Evaluation
    eval_episodes: int = 100
    checkpoint_frequency: int = 100
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = TrainingConfig()