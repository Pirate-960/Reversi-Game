import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import copy
import time
import os
import psutil
import traceback
from datetime import datetime, timedelta
from OthelloGameEngine import Othello, Player, AdvancedOthelloAI
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
import threading
from concurrent.futures import ThreadPoolExecutor

# ======================
# ENHANCED CONFIGURATION
# ======================
LOG_LEVEL = "DEBUG"
LOG_FILE = f"othello_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
PRINT_PROGRESS = True
PERF_LOG_INTERVAL = 300
DETAILED_MCTS_LOGGING = True

# Game Configuration
BOARD_SIZE = 8
NUM_CHANNELS = 3
NUM_RES_BLOCKS = 10  # Increased from 5
MCTS_SIMULATIONS = 1600  # Increased from 800
BUFFER_SIZE = 1000000
BATCH_SIZE = 2048  # Increased from 1024
NUM_EPISODES = 1000  # Increased from 5
EVAL_INTERVAL = 20  # More frequent evaluation
CHECKPOINT_INTERVAL = 100

# Training Hyperparameters
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
TEMPERATURE_INIT = 1.0
TEMPERATURE_FINAL = 0.1
TEMPERATURE_DECAY = 0.97
PUCT_BASE = 19652
PUCT_INIT = 1.25
MAX_WORKERS = 4

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
SCALER = GradScaler()

Transition = namedtuple('Transition', ('state', 'policy', 'value', 'priority'))

class Logger:
    """Advanced logging system with performance monitoring"""
    def __init__(self):
        self.start_time = time.time()
        self.last_perf_log = self.start_time
        self.file = open(LOG_FILE, 'a') if LOG_FILE else None
        self.process = psutil.Process()
        self.gpu_available = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
        
    def _log(self, level, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        if PRINT_PROGRESS and level in ["INFO", "WARNING", "ERROR"]:
            print(log_entry)
            
        if self.file:
            self.file.write(log_entry + "\n")
            self.file.flush()
            
    def log(self, level, message):
        """Main logging method with level filtering"""
        if LOG_LEVEL == "DEBUG":
            self._log(level, message)
        elif LOG_LEVEL == "INFO" and level in ["INFO", "WARNING", "ERROR"]:
            self._log(level, message)
        elif LOG_LEVEL == "WARNING" and level in ["WARNING", "ERROR"]:
            self._log(level, message)
        elif LOG_LEVEL == "ERROR" and level == "ERROR":
            self._log(level, message)
            
    def perf_stats(self):
        """Log detailed system performance metrics"""
        stats = {
            "elapsed": time.time() - self.start_time,
            "cpu_usage": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss // 1024 // 1024,
            "gpu_mem_used": torch.xpu.memory_allocated() // 1024 // 1024 if self.gpu_available else 0,
            "gpu_mem_total": torch.xpu.get_device_properties(0).total_memory // 1024 // 1024 if self.gpu_available else 0,
            "threads": self.process.num_threads()
        }
        self._log("PERF", f"Performance|{stats}")
        
    def close(self):
        """Cleanup resources"""
        if self.file:
            self.file.close()

logger = Logger()

class Chrono:
    """High precision timing context manager"""
    def __init__(self, name):
        self.name = name
        self.start = None
        self.duration = None
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.perf_counter() - self.start
        logger.log("DEBUG", f"Timing|{self.name}|{self.duration:.4f}s")

# ======================
# ENHANCED NEURAL NETWORK
# ======================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """Enhanced Residual block with SE attention"""
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = self.dropout(x)
        return torch.relu(x + residual)

class OthelloNet(nn.Module):
    """Enhanced dual-headed neural network"""
    def __init__(self):
        super().__init__()
        # Shared trunk with increased channels
        self.conv = nn.Conv2d(NUM_CHANNELS, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(NUM_RES_BLOCKS)])
        
        # Policy head with attention
        self.policy_conv = nn.Conv2d(256, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_se = SEBlock(32)
        self.policy_fc = nn.Linear(32*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        
        # Value head with deeper network
        self.value_conv = nn.Conv2d(256, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_se = SEBlock(32)
        self.value_fc1 = nn.Linear(32*BOARD_SIZE*BOARD_SIZE, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Shared features
        x = torch.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # Policy head
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_se(p)
        p = self.policy_fc(p.view(-1, 32*BOARD_SIZE*BOARD_SIZE))
        policy = torch.softmax(p, dim=1)
        
        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = self.value_se(v)
        v = torch.relu(self.value_fc1(v.view(-1, 32*BOARD_SIZE*BOARD_SIZE)))
        v = torch.relu(self.value_fc2(v))
        value = torch.tanh(self.value_fc3(v))
        
        return policy, value

# ======================
# ENHANCED MCTS
# ======================
class MCTSNode:
    """Enhanced MCTS node with parallel processing support"""
    def __init__(self, game_state, player, parent=None, prior=0.0):
        self.game = game_state
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior = prior
        self.value_sum = 0.0
        self.value_square_sum = 0.0
        
    def ucb_score(self, total_visits, puct_c):
        """Enhanced UCB score with uncertainty estimation"""
        if self.visits == 0:
            return float('inf')
        
        # Q-value with uncertainty bonus
        q_value = self.total_value / self.visits
        uncertainty = np.sqrt(
            max(0, self.value_square_sum/self.visits - (self.value_sum/self.visits)**2)
        )
        
        # UCB formula with PUCT
        exploit = q_value + uncertainty
        explore = puct_c * self.prior * np.sqrt(total_visits) / (1 + self.visits)
        
        return exploit + explore

    def select_child(self, temperature=1.0):
        """Temperature-based selection"""
        visits = np.array([child.visits for child in self.children.values()])
        if temperature == 0:
            action_idx = np.argmax(visits)
        else:
            # Apply temperature
            visits = visits ** (1/temperature)
            visits = visits / visits.sum()
            action_idx = np.random.choice(len(visits), p=visits)
        
        moves = list(self.children.keys())
        return moves[action_idx], list(self.children.values())[action_idx]

class ParallelMCTS:
    """MCTS with parallel simulation support"""
    def __init__(self, net, num_simulations, num_workers=MAX_WORKERS):
        self.net = net
        self.num_simulations = num_simulations
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
    def simulate(self, state, node):
        """Single MCTS simulation"""
        path = []
        current_node = node
        
        # Selection
        while not current_node.is_leaf():
            path.append(current_node)
            _, current_node = current_node.select_child()
            
        # Expansion and evaluation
        if not current_node.game.is_game_over():
            with torch.no_grad():
                policy, value = self.net(state)
                current_node.expand(policy[0].cpu().numpy())
                
            path.append(current_node)
            value = value.item()
        else:
            value = current_node.game.get_winner()
            
        # Backpropagation with value statistics
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            node.value_square_sum += value * value
            value = -value
            
    def run(self, root_state, root_node):
        """Run parallel MCTS simulations"""
        futures = []
        for _ in range(self.num_simulations):
            futures.append(
                self.executor.submit(self.simulate, root_state, root_node)
            )
        
        # Wait for all simulations to complete
        for future in futures:
            future.result()
            
        return root_node

class AlphaZeroAgent:
    """Enhanced reinforcement learning agent"""
    def __init__(self, player):
        self.net = OthelloNet().to(DEVICE)
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=NUM_EPISODES,
            eta_min=MIN_LEARNING_RATE
        )
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.player = player
        self.mcts = ParallelMCTS(self.net, MCTS_SIMULATIONS)

    def state_to_tensor(self, game):
        """Convert game state to neural network input tensor"""
        board = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        
        # Channel 0: Current player's pieces
        # Channel 1: Opponent's pieces
        # Channel 2: Current player indicator
        current_player = game.current_player
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == current_player.value:
                    board[0][r][c] = 1.0
                elif game.board[r][c] != ".":
                    board[1][r][c] = 1.0
        
        # Fill channel 2 with 1s if current player is black, 0s if white
        board[2].fill(1.0 if current_player == Player.BLACK else 0.0)
        
        return torch.tensor(board, dtype=torch.float32).unsqueeze(0)

    def augment_state(self, state):
        """Data augmentation through rotations and flips"""
        state = state.squeeze(0).cpu().numpy()  # Convert tensor to numpy array
        augmented_states = []
        for k in range(4):  # 4 rotations
            rotated = np.rot90(state, k=k)
            augmented_states.append(torch.tensor(rotated).unsqueeze(0))
            augmented_states.append(torch.tensor(np.fliplr(rotated)).unsqueeze(0))
        return augmented_states
    
    def train_step(self, batch):
        """Enhanced training step with mixed precision"""
        states, policies, values, priorities = zip(*batch)
        states = torch.cat(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(DEVICE)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(DEVICE)
        priorities = torch.tensor(np.array(priorities), dtype=torch.float32).to(DEVICE)
        
        self.optimizer.zero_grad()
        
        with autocast():
            pred_policies, pred_values = self.net(states)
            
            # Weighted policy loss
            policy_loss = -(priorities * policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
            
            # Value loss with Huber loss
            value_loss = F.smooth_l1_loss(pred_values.squeeze(), values)
            
            # Combined loss
            loss = policy_loss + value_loss
        
        # Scaled backward pass
        SCALER.scale(loss).backward()
        SCALER.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        SCALER.step(self.optimizer)
        SCALER.update()
        
        self.scheduler.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()

class RLTrainingManager:
    """Enhanced training management system"""
    def __init__(self):
        self.agent = AlphaZeroAgent(Player.BLACK)
        # Multiple opponent strengths with level attribute
        self.opponents = [
            self.create_opponent(3),
            self.create_opponent(5),
            self.create_opponent(8),
            self.create_opponent(10)
        ]
        self.temperature = TEMPERATURE_INIT
        self.start_time = time.time()
        self.episode_times = []
        self.best_win_rate = -np.inf

    def create_opponent(self, level):
        """Create an opponent with a specified level"""
        opponent = AdvancedOthelloAI(Player.WHITE, level)
        opponent.level = level  # Add level attribute
        return opponent
        
    def train_cycle(self):
        """Enhanced training cycle with curriculum learning"""
        try:
            for episode in range(NUM_EPISODES):
                # Temperature annealing
                self.temperature = max(
                    TEMPERATURE_FINAL,
                    TEMPERATURE_INIT * (TEMPERATURE_DECAY ** episode)
                )
                
                # Generate self-play data with augmentation
                game_data = self.self_play_game(self.temperature)
                augmented_data = []
                for transition in game_data:
                    states = self.agent.augment_state(transition.state)
                    for aug_state in states:
                        augmented_data.append(
                            Transition(aug_state, transition.policy,
                                     transition.value, transition.priority)
                        )
                self.agent.replay_buffer.extend(augmented_data)
                
                # Training with prioritized sampling
                if len(self.agent.replay_buffer) >= BATCH_SIZE:
                    priorities = np.array([t.priority for t in self.agent.replay_buffer])
                    probs = priorities / priorities.sum()
                    indices = np.random.choice(
                        len(self.agent.replay_buffer),
                        BATCH_SIZE,
                        p=probs
                    )
                    batch = [self.agent.replay_buffer[i] for i in indices]
                    loss, policy_loss, value_loss = self.agent.train_step(batch)
                    
                # Evaluation and checkpointing
                if (episode + 1) % EVAL_INTERVAL == 0:
                    # Evaluate against multiple opponents
                    win_rates = []
                    for opponent in self.opponents:
                        win_rate = self.evaluate(opponent)
                        win_rates.append(win_rate)
                    
                    avg_win_rate = np.mean(win_rates)
                    if avg_win_rate > self.best_win_rate:
                        self.best_win_rate = avg_win_rate
                        self.agent.save_checkpoint("best_model.pth")
                
                if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                    self.agent.save_checkpoint(f"checkpoint_ep{episode+1}.pth")
                
                # Progress logging
                logger.log("INFO",
                    f"Episode:{episode+1}|Temp:{self.temperature:.2f}|"
                    f"Loss:{loss:.4f}|PolicyLoss:{policy_loss:.4f}|"
                    f"ValueLoss:{value_loss:.4f}")
                
                # Performance monitoring
                if time.time() - logger.last_perf_log >= PERF_LOG_INTERVAL:
                    logger.perf_stats()
                    
        except Exception as e:
            logger.log("ERROR", f"Training failed: {str(e)}\n{traceback.format_exc()}")
            raise
        finally:
            self.cleanup()
            
    def self_play_game(self, temperature):
        """Enhanced self-play with temperature-based exploration"""
        game = Othello()
        history = []
        current_player = Player.BLACK
        move_count = 0
        
        while not game.is_game_over():
            valid_moves = game.get_valid_moves(current_player)
            if not valid_moves:
                current_player = current_player.opponent()
                continue
                
            # Create state tensor
            state_tensor = self.agent.state_to_tensor(game).to(DEVICE)
            
            # MCTS search with current temperature
            root = MCTSNode(game, current_player)
            self.agent.mcts.run(state_tensor, root)
            
            # Calculate visit count policy
            policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
            total_visits = sum(child.visits for child in root.children.values())
            for move, child in root.children.items():
                policy[move[0] * BOARD_SIZE + move[1]] = child.visits / total_visits
                
            # Temperature-based move selection
            if temperature == 0:
                move = max(root.children.items(), key=lambda x: x[1].visits)[0]
            else:
                move, _ = root.select_child(temperature)
                
            # Store transition with priority
            priority = root.children[move].visits / total_visits
            history.append(Transition(
                state_tensor.cpu(),
                policy,
                None,  # Value will be updated later
                priority
            ))
            
            # Apply move
            game.apply_move(move[0], move[1], current_player)
            current_player = current_player.opponent()
            move_count += 1
            
            logger.log("DEBUG",
                f"SelfPlay|Move:{move_count}|"
                f"Player:{current_player.value}|"
                f"Temperature:{temperature:.2f}")
                
        # Update values based on game outcome
        final_value = game.get_winner()
        for idx, transition in enumerate(history):
            value = final_value if idx % 2 == 0 else -final_value
            history[idx] = transition._replace(value=value)
            
        return history
        
    def evaluate(self, opponent, num_games=20):
        """Enhanced evaluation against specific opponent"""
        wins = draws = 0
        self.agent.net.eval()  # Set to evaluation mode
        
        logger.log("INFO",
            f"Evaluation|Start|"
            f"Opponent:Level{opponent.level}")
            
        for game_num in range(num_games):
            game = Othello()
            current_player = Player.BLACK
            
            while not game.is_game_over():
                if current_player == self.agent.player:
                    # Agent move with temperature=0 (deterministic)
                    root = MCTSNode(game, current_player)
                    state_tensor = self.agent.state_to_tensor(game).to(DEVICE)
                    self.agent.mcts.run(state_tensor, root)
                    move = max(root.children.items(), key=lambda x: x[1].visits)[0]
                else:
                    # Opponent move
                    move = opponent.find_best_move(game)
                    
                game.apply_move(move[0], move[1], current_player)
                current_player = current_player.opponent()
                
            # Game result
            winner = game.get_winner()
            if winner == 0:
                draws += 1
            elif (winner == 1 and self.agent.player == Player.BLACK) or \
                 (winner == -1 and self.agent.player == Player.WHITE):
                wins += 1
                
            logger.log("DEBUG",
                f"Eval|Game:{game_num+1}|"
                f"Winner:{winner}|"
                f"Moves:{game.move_count}")
                
        self.agent.net.train()  # Back to training mode
        win_rate = (wins + 0.5 * draws) / num_games * 100
        
        logger.log("INFO",
            f"Evaluation|Complete|"
            f"WinRate:{win_rate:.1f}%|"
            f"Wins:{wins}|Draws:{draws}")
            
        return win_rate
        
    def cleanup(self):
        """Cleanup resources"""
        self.agent.mcts.executor.shutdown()
        logger.close()
        
if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create training manager and start training
    trainer = RLTrainingManager()
    
    try:
        trainer.train_cycle()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
    finally:
        # Final evaluation
        final_win_rate = trainer.evaluate(
            trainer.opponents[-1],  # Strongest opponent
            num_games=100
        )
        logger.log("INFO", f"Final|WinRate:{final_win_rate:.1f}%")