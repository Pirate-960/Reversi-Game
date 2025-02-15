import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from collections import deque, namedtuple
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import ray
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from OthelloGameEngine import Othello, Player, AdvancedOthelloAI

# Enhanced Configuration
BOARD_SIZE = 8
NUM_CHANNELS = 3
NUM_RES_BLOCKS = 19  # Increased from 5
MCTS_SIMULATIONS = 1600  # Doubled
BUFFER_SIZE = 500000  # Increased
BATCH_SIZE = 2048  # Increased
NUM_EPISODES = 5000
INITIAL_TEMP = 1.0
MIN_TEMP = 0.1
TEMP_DECAY = 0.97
C_PUCT = 4.0  # MCTS exploration constant
VALUE_WEIGHT = 1.0
POLICY_WEIGHT = 1.5
L2_WEIGHT = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'policy', 'value'))

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e.expand_as(x)

class EnhancedResBlock(nn.Module):
    """Enhanced Residual block with SE attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return torch.relu(x)

class EnhancedOthelloNet(nn.Module):
    """Enhanced Neural Network with SE blocks and deeper architecture"""
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([EnhancedResBlock(256) for _ in range(NUM_RES_BLOCKS)])
        
        # Enhanced Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        )
        
        # Enhanced Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv_input(x)
        
        # Apply residual blocks with skip connections
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = self.policy_head(x)
        policy = torch.softmax(policy, dim=1)
        
        # Value head
        value = self.value_head(x)
        
        return policy, value

@ray.remote
class SelfPlayWorker:
    """Distributed self-play worker"""
    def __init__(self, network_weights, player):
        self.net = EnhancedOthelloNet().to(DEVICE)
        self.net.load_state_dict(network_weights)
        self.net.eval()
        self.player = player
        
    def generate_game(self, temperature):
        game = Othello()
        history = []
        
        with torch.no_grad():
            while not game.is_game_over():
                state_tensor = self.state_to_tensor(game)
                policy, _ = self.mcts_search(game, temperature)
                
                valid_moves = game.get_valid_moves(game.current_player)
                move_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
                for move in valid_moves:
                    idx = move[0] * BOARD_SIZE + move[1]
                    move_probs[idx] = policy[idx]
                
                # Temperature-adjusted move selection
                move_probs = move_probs ** (1 / temperature)
                move_probs /= move_probs.sum()
                
                history.append((state_tensor, move_probs))
                
                chosen_move_idx = np.random.choice(len(valid_moves), p=move_probs)
                chosen_move = valid_moves[chosen_move_idx]
                game.apply_move(chosen_move[0], chosen_move[1], game.current_player)
                game.switch_player()
        
        return self.process_game_history(game, history)

class EnhancedAlphaZeroAgent:
    """Enhanced RL Agent with distributed training"""
    def __init__(self, player):
        self.net = EnhancedOthelloNet().to(DEVICE)
        self.net = DDP(self.net) if torch.cuda.device_count() > 1 else self.net
        
        # Enhanced optimizer setup
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=0.001,
            weight_decay=L2_WEIGHT,
            betas=(0.9, 0.999)
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPISODES)
        self.scaler = GradScaler()  # For mixed precision training
        
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.player = player
        
        # Initialize wandb logging
        wandb.init(project="alphazero-othello", config={
            "num_res_blocks": NUM_RES_BLOCKS,
            "mcts_sims": MCTS_SIMULATIONS,
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE
        })
        
    def train_step(self, batch):
        """Enhanced training step with mixed precision"""
        states, policies, values = zip(*batch)
        states = torch.cat(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(DEVICE)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(DEVICE)
        
        self.optimizer.zero_grad()
        
        with autocast():
            pred_policies, pred_values = self.net(states)
            
            # Enhanced loss calculation
            policy_loss = -(policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(pred_values.squeeze(), values)
            total_loss = POLICY_WEIGHT * policy_loss + VALUE_WEIGHT * value_loss
        
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        wandb.log({
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        })
        
        return total_loss.item()

class EnhancedTrainingManager:
    """Enhanced training manager with distributed processing"""
    def __init__(self):
        self.agent = EnhancedAlphaZeroAgent(Player.BLACK)
        ray.init()
        self.num_workers = mp.cpu_count()
        self.workers = [
            SelfPlayWorker.remote(self.agent.net.state_dict(), Player.BLACK)
            for _ in range(self.num_workers)
        ]
        
    def train_cycle(self):
        """Enhanced training loop with parallel self-play"""
        temperature = INITIAL_TEMP
        
        for episode in range(NUM_EPISODES):
            # Distributed self-play
            game_futures = [
                worker.generate_game.remote(temperature)
                for worker in self.workers
            ]
            games_data = ray.get(game_futures)
            
            # Process collected data
            for game_data in games_data:
                self.agent.replay_buffer.extend(game_data)
            
            # Training
            if len(self.agent.replay_buffer) >= BATCH_SIZE:
                batch = random.sample(self.agent.replay_buffer, BATCH_SIZE)
                loss = self.agent.train_step(batch)
                
                # Update temperature
                temperature = max(MIN_TEMP, temperature * TEMP_DECAY)
                
                wandb.log({
                    "episode": episode,
                    "temperature": temperature,
                    "buffer_size": len(self.agent.replay_buffer)
                })
            
            # Periodic evaluation and checkpointing
            if (episode + 1) % 100 == 0:
                self.evaluate_and_save(episode)
                
            self.agent.scheduler.step()
    
    def evaluate_and_save(self, episode):
        """Enhanced evaluation against multiple opponents"""
        win_rates = self.parallel_evaluation()
        wandb.log({
            "average_win_rate": np.mean(win_rates),
            "max_win_rate": np.max(win_rates)
        })
        
        # Save checkpoint with metadata
        torch.save({
            'episode': episode,
            'model_state_dict': self.agent.net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'scheduler_state_dict': self.agent.scheduler.state_dict(),
            'win_rates': win_rates,
        }, f"checkpoint_episode_{episode}.pt")

if __name__ == "__main__":
    manager = EnhancedTrainingManager()
    manager.train_cycle()