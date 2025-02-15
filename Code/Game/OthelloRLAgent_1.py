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

# ======================
# LOGGING CONFIGURATION
# ======================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = f"othello_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
PRINT_PROGRESS = True
PERF_LOG_INTERVAL = 300  # Seconds between performance snapshots
DETAILED_MCTS_LOGGING = True  # Enable for per-simulation logging

# ======================
# GAME CONFIGURATION
# ======================
BOARD_SIZE = 8
NUM_CHANNELS = 3  # Current player, Black pieces, White pieces
NUM_RES_BLOCKS = 5
MCTS_SIMULATIONS = 800
BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
NUM_EPISODES = 1000
EVAL_INTERVAL = 50  # Episodes between evaluations
CHECKPOINT_INTERVAL = 100  # Episodes between checkpoints

# ======================
# SYSTEM CONFIGURATION
# ======================
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
torch.set_float32_matmul_precision('high')  # For Intel Arc optimization

Transition = namedtuple('Transition', ('state', 'policy', 'value'))

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
# NEURAL NETWORK ARCH
# ======================
class ResBlock(nn.Module):
    """Residual block with spectral normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)

class OthelloNet(nn.Module):
    """Dual-headed neural network with policy and value heads"""
    def __init__(self):
        super().__init__()
        # Shared trunk
        self.conv = nn.Conv2d(NUM_CHANNELS, 128, 3, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(NUM_RES_BLOCKS)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE*BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Shared features
        x = torch.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        
        # Policy head
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.view(-1, 2*BOARD_SIZE*BOARD_SIZE))
        policy = torch.softmax(p, dim=1)
        
        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = torch.relu(self.value_fc1(v.view(-1, BOARD_SIZE*BOARD_SIZE)))
        value = torch.tanh(self.value_fc2(v))
        
        logger.log("DEBUG", 
            f"Network|PolicySum:{policy.sum().item():.2f}|"
            f"Value:{value.item():.2f}")
            
        return policy, value

# ======================
# MCTS IMPLEMENTATION
# ======================
class MCTSNode:
    """Monte Carlo Tree Search node with enhanced statistics"""
    def __init__(self, game_state, player, parent=None):
        self.game = game_state
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior = 0.0
        self.expansion_time = 0.0
        
    def is_leaf(self):
        return not self.children
    
    def expand(self, policy_probs):
        """Expand node with possible moves and policy priors"""
        valid_moves = self.game.get_valid_moves(self.player)
        if not valid_moves:
            logger.log("DEBUG", "MCTS|No valid moves for expansion")
            return
            
        total_prior = sum(policy_probs[m[0]*8 + m[1]] for m in valid_moves)
        for move in valid_moves:
            new_game = copy.deepcopy(self.game)
            new_game.apply_move(move[0], move[1], self.player)
            child = MCTSNode(new_game, self.player.opponent(), parent=self)
            prior = policy_probs[move[0]*8 + move[1]] / total_prior
            child.prior = prior
            self.children[move] = child
            if DETAILED_MCTS_LOGGING:
                logger.log("DEBUG",
                    f"MCTS|Expand|Move:{move}|Prior:{prior:.3f}")

        logger.log("DEBUG",
            f"MCTS|NodeExpanded|Children:{len(self.children)}|"
            f"TotalPriors:{total_prior:.3f}")
            
    def select_child(self):
        """Select child using PUCT algorithm with exploration factor"""
        total_visits = sum(child.visits for child in self.children.values())
        sqrt_total = np.sqrt(total_visits + 1e-8)
        
        best_score = -np.inf
        best_move = None
        best_child = None
        
        for move, child in self.children.items():
            exploit = child.total_value / (child.visits + 1e-8)
            explore = child.prior * sqrt_total / (child.visits + 1)
            score = exploit + 2.0 * explore
            
            if DETAILED_MCTS_LOGGING:
                logger.log("DEBUG",
                    f"MCTS|Select|Move:{move}|"
                    f"Visits:{child.visits}|Value:{child.total_value:.2f}|"
                    f"Exploit:{exploit:.2f}|Explore:{explore:.2f}|Score:{score:.2f}")

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
                
        return best_move, best_child

class AlphaZeroAgent:
    """Reinforcement learning agent with instrumentation"""
    def __init__(self, player):
        self.net = OthelloNet().to(DEVICE)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=0.001, weight_decay=1e-4)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.player = player
        self.mcts_stats = {
            'total_simulations': 0,
            'max_depth': 0,
            'total_expansions': 0
        }
        
        param_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logger.log("INFO",
            f"Network|Params:{param_count}|"
            f"ResBlocks:{NUM_RES_BLOCKS}|Device:{DEVICE}")

    def state_to_tensor(self, game):
        """Convert game state to neural network input tensor"""
        board = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        current_player = 1 if game.current_player == self.player else 0
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == self.player.value:
                    board[1][r][c] = 1.0
                elif game.board[r][c] != ".":
                    board[2][r][c] = 1.0
        board[0].fill(current_player)
        
        tensor = torch.tensor(board).unsqueeze(0).to(DEVICE)
        logger.log("DEBUG", f"Preprocess|TensorShape:{tensor.shape}")
        return tensor
    
    def mcts_search(self, game):
        """Perform full MCTS search with detailed logging"""
        root = MCTSNode(game, self.player)
        logger.log("DEBUG", f"MCTS|Root|Player:{self.player.value}")
        
        # Initial expansion
        with Chrono("MCTS_ROOT_EXPANSION"):
            with torch.no_grad():
                state_tensor = self.state_to_tensor(root.game)
                policy, _ = self.net(state_tensor)
                policy_probs = policy.squeeze().cpu().numpy()
                root.expand(policy_probs)
                logger.log("DEBUG",
                    f"MCTS|RootPolicy|Max:{policy_probs.max():.3f}|"
                    f"Min:{policy_probs.min():.3f}")

        # MCTS simulations
        for sim in range(MCTS_SIMULATIONS):
            node = root
            search_path = [node]
            current_depth = 0
            
            # Selection phase
            while not node.is_leaf():
                move, node = node.select_child()
                search_path.append(node)
                current_depth += 1
                self.mcts_stats['total_expansions'] += 1

            # Expansion and evaluation
            value = 0
            if not node.game.is_game_over():
                with Chrono("MCTS_NN_EVALUATION"):
                    with torch.no_grad():
                        state_tensor = self.state_to_tensor(node.game)
                        child_policy, value = self.net(state_tensor)
                        policy_probs = child_policy.squeeze().cpu().numpy()
                        node.expand(policy_probs)
                        value = value.item()
                        logger.log("DEBUG",
                            f"MCTS|LeafEval|Value:{value:.2f}|"
                            f"PolicySum:{policy_probs.sum():.2f}")
            else:
                x, o = node.game.count_discs()
                value = 1 if (x > o and self.player == Player.BLACK) else -1
                logger.log("DEBUG",
                    f"MCTS|Terminal|X:{x} O:{o}|Value:{value}")

            # Backpropagation
            for node in reversed(search_path):
                node.visits += 1
                node.total_value += value
                value = -value  # Switch perspective
                if DETAILED_MCTS_LOGGING:
                    logger.log("DEBUG",
                        f"MCTS|Backprop|Visits:{node.visits}|"
                        f"TotalValue:{node.total_value:.2f}")

            # Update statistics
            self.mcts_stats['total_simulations'] += 1
            self.mcts_stats['max_depth'] = max(self.mcts_stats['max_depth'], current_depth)
            
        # Generate visit counts
        visit_counts = np.zeros(64)
        total_visits = sum(child.visits for child in root.children.values())
        for move, child in root.children.items():
            idx = move[0] * 8 + move[1]
            visit_counts[idx] = child.visits / total_visits
            
        logger.log("INFO",
            f"MCTS|Summary|Sims:{MCTS_SIMULATIONS}|"
            f"MaxDepth:{self.mcts_stats['max_depth']}|"
            f"Expansions:{self.mcts_stats['total_expansions']}")
            
        return visit_counts, root.total_value / root.visits
    
    def train(self, batch):
        """Training step with gradient clipping and logging"""
        states, policies, values = zip(*batch)
        states = torch.cat(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(DEVICE)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(DEVICE)
        
        self.optimizer.zero_grad()
        pred_policies, pred_values = self.net(states)
        
        # Policy loss
        policy_loss = -(policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
        
        # Value loss
        value_loss = torch.nn.functional.mse_loss(pred_values.squeeze(), values)
        
        # Combine losses
        loss = policy_loss + value_loss
        
        # Backward pass with gradient clipping
        with Chrono("BACKWARD_PASS"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            self.optimizer.step()
        
        # Log gradients
        grad_norms = [p.grad.norm().item() for p in self.net.parameters() if p.grad is not None]
        logger.log("DEBUG",
            f"Training|PolicyLoss:{policy_loss.item():.4f}|"
            f"ValueLoss:{value_loss.item():.4f}|"
            f"GradAvg:{sum(grad_norms)/len(grad_norms):.4f}|"
            f"GradMax:{max(grad_norms):.4f}")
            
        return loss.item()
    
    def save_checkpoint(self, path):
        """Save model checkpoint with metadata"""
        state = {
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'mcts_stats': self.mcts_stats,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(state, path)
        logger.log("INFO", f"Checkpoint|Saved:{path}")
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        if os.path.exists(path):
            state = torch.load(path, map_location=DEVICE)
            self.net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.mcts_stats = state.get('mcts_stats', self.mcts_stats)
            logger.log("INFO", f"Checkpoint|Loaded:{path}")

# ======================
# TRAINING FRAMEWORK
# ======================
class RLTrainingManager:
    """Complete training management system"""
    def __init__(self):
        self.agent = AlphaZeroAgent(Player.BLACK)
        self.opponents = [
            AdvancedOthelloAI(Player.WHITE, 3),
            AdvancedOthelloAI(Player.WHITE, 5),
            AdvancedOthelloAI(Player.WHITE, 8)
        ]
        self.start_time = time.time()
        self.episode_times = []
        self.best_win_rate = -np.inf
        
    def _time_remaining(self, completed, total):
        if len(self.episode_times) < 1:
            return "N/A"
        avg_time = np.mean(self.episode_times)
        remaining = (total - completed) * avg_time
        return str(timedelta(seconds=int(remaining)))
        
    def self_play_game(self):
        """Generate self-play data with fixed-size policy vectors"""
        game = Othello()
        history = []
        move_count = 0
    
        logger.log("INFO", "SelfPlay|GameStart")
        with Chrono("SELF_PLAY_GAME"):
            while not game.is_game_over():
                valid_moves = game.get_valid_moves(game.current_player)
                if not valid_moves:
                    game.switch_player()
                    logger.log("DEBUG", f"SelfPlay|Pass|Player:{game.current_player.value}")
                    continue

                with Chrono("MCTS_SEARCH"):
                    policy, value = self.agent.mcts_search(game)

                # Create full 64-element policy vector
                full_policy = np.zeros(64, dtype=np.float32)
                valid_indices = [m[0]*8 + m[1] for m in valid_moves]
                total = sum(policy[i] for i in valid_indices)
            
                if total == 0:
                    full_policy[valid_indices] = 1.0 / len(valid_indices)
                    logger.log("DEBUG", "SelfPlay|UniformPolicy")
                else:
                    full_policy[valid_indices] = policy[valid_indices] / total

                history.append(Transition(
                    self.agent.state_to_tensor(game),
                    full_policy,
                    None
                ))

                chosen_idx = np.random.choice(len(valid_moves), p=full_policy[valid_indices])
                chosen_move = valid_moves[chosen_idx]
                game.apply_move(chosen_move[0], chosen_move[1], game.current_player)
                game.switch_player()
            
                logger.log("DEBUG", 
                    f"SelfPlay|Move:{move_count}|Player:{game.current_player.value}|"
                    f"Chosen:{chosen_move}|ValueEst:{value:.2f}|"
                    f"Policy:{[f'{p:.2f}' for p in full_policy[valid_indices]]}")
                move_count += 1

            x, o = game.count_discs()
            final_value = 1 if (x > o and self.agent.player == Player.BLACK) else -1
            for i in range(len(history)):
                history[i] = history[i]._replace(value=final_value * (-1)**i)
            
            logger.log("INFO",
                f"SelfPlay|Completed|Moves:{move_count}|"
                f"Score:X:{x} O:{o}|FinalValue:{final_value}")
            logger.log("DEBUG", f"SelfPlay|FinalBoard:\n{game.board}")
    
        return history
    
    def evaluate(self, num_games=20):
        """Evaluate agent against current opponent"""
        wins = 0
        opponent = self.opponents[1]
        
        logger.log("INFO", f"Eval|Start|OpponentLevel:{opponent.search_depth}")
        with Chrono("EVALUATION_PHASE"):
            for game_num in range(num_games):
                game = Othello()
                while not game.is_game_over():
                    if game.current_player == self.agent.player:
                        policy, _ = self.agent.mcts_search(game)
                        valid_moves = game.get_valid_moves(game.current_player)
                        move_idx = np.argmax([policy[m[0]*8 + m[1]] for m in valid_moves])
                        move = valid_moves[move_idx]
                    else:
                        move = opponent.find_best_move(game)
                        
                    game.apply_move(move[0], move[1], game.current_player)
                    game.switch_player()
                    
                x, o = game.count_discs()
                result = "Win" if ((x > o and self.agent.player == Player.BLACK) or 
                                  (o > x and self.agent.player == Player.WHITE)) else "Loss"
                wins += 1 if result == "Win" else 0
                logger.log("DEBUG",
                    f"Eval|Game:{game_num+1}|Result:{result}|X:{x} O:{o}")

            win_rate = (wins / num_games) * 100
            logger.log("INFO",
                f"Eval|Complete|Games:{num_games}|"
                f"Wins:{wins}|WinRate:{win_rate:.1f}%")
            return win_rate
    
    def train_cycle(self, num_episodes=NUM_EPISODES):
        """Complete training cycle with progress tracking"""
        logger.log("INFO", 
            f"Training|Start|Episodes:{num_episodes}|"
            f"BatchSize:{BATCH_SIZE}|BufferSize:{BUFFER_SIZE}")
        start_time = time.time()
        
        try:
            for episode in range(num_episodes):
                ep_start = time.time()
                logger.log("INFO", f"Episode|Start:{episode+1}/{num_episodes}")
                
                # Generate self-play data
                with Chrono("SELF_PLAY_DATA_GEN"):
                    game_data = self.self_play_game()
                    self.agent.replay_buffer.extend(game_data)
                    logger.log("DEBUG",
                        f"Buffer|Added:{len(game_data)}|"
                        f"Total:{len(self.agent.replay_buffer)}")
                    
                # Training step
                if len(self.agent.replay_buffer) >= BATCH_SIZE:
                    with Chrono("TRAINING_STEP"):
                        batch = random.sample(self.agent.replay_buffer, BATCH_SIZE)
                        values = [t.value for t in batch]
                        logger.log("DEBUG",
                            f"Training|Batch|Values|"
                            f"Mean:{np.mean(values):.2f}|Std:{np.std(values):.2f}")
                        loss = self.agent.train(batch)
                        logger.log("INFO", f"Training|Loss:{loss:.4f}")
                        
                # Performance logging
                if time.time() - logger.last_perf_log >= PERF_LOG_INTERVAL:
                    logger.perf_stats()
                    
                # Checkpoint and evaluation
                if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                    self.agent.save_checkpoint(f"checkpoint_ep{episode+1}.pth")
                    
                if (episode + 1) % EVAL_INTERVAL == 0:
                    win_rate = self.evaluate()
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        self.agent.save_checkpoint("best_model.pth")
                        logger.log("INFO",
                            f"Checkpoint|NewBest|WinRate:{win_rate:.1f}%")
                        
                # Timing statistics
                ep_time = time.time() - ep_start
                self.episode_times.append(ep_time)
                avg_time = np.mean(self.episode_times[-10:])
                remaining = self._time_remaining(episode+1, num_episodes)
                
                logger.log("INFO",
                    f"Episode|Complete:{episode+1}|"
                    f"Time:{ep_time:.1f}s|Avg:{avg_time:.1f}s|"
                    f"ETA:{remaining}")
                
        except Exception as e:
            logger.log("ERROR", 
                f"Training|Failed:{str(e)}\n"
                f"Traceback:{traceback.format_exc()}")
            raise
        finally:
            total_time = time.time() - start_time
            logger.log("INFO",
                f"Training|Completed|Time:{timedelta(seconds=int(total_time))}|"
                f"AvgEpisode:{np.mean(self.episode_times):.1f}s")
            logger.close()

if __name__ == "__main__":
    trainer = RLTrainingManager()
    trainer.train_cycle()
    
    final_win_rate = trainer.evaluate(num_games=100)
    logger.log("INFO", f"Final|WinRate:{final_win_rate:.1f}%")