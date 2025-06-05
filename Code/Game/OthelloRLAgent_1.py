"""
Advanced AlphaZero implementation for Othello using a hybrid neural network with CNN, Transformer, and GNN components.
Integrates Monte Carlo Tree Search (MCTS) with prioritized experience replay, distributed training, and experiment tracking.
Requires OthelloGameEngine module and dependencies: torch, numpy, psutil, wandb, torch-geometric.

Usage:
    python advanced_alphazero.py
    Ensure WandB is configured for experiment tracking (optional).
    Checkpoints and logs are saved in the working directory.
"""

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
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, DataLoader
from OthelloGameEngine import Othello, Player, AdvancedOthelloAI
import logging
import wandb
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GCNConv
import math
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# ======================
# ENHANCED CONFIGURATION
# ======================
BOARD_SIZE = 8  # Size of the Othello board (8x8)
NUM_CHANNELS = 13  # Input channels: player, pieces, move history, legal moves
NUM_RES_BLOCKS = 14  # Number of ResNeXt blocks in neural network
ATTENTION_HEADS = 12  # Number of attention heads in Transformer
TRANSFORMER_LAYERS = 3  # Number of Transformer layers
GNN_LAYERS = 2  # Number of graph neural network layers
MCTS_SIMULATIONS = 2000  # Number of MCTS simulations per move
MCTS_TEMP_INIT = 1.2  # Initial MCTS temperature for exploration
MCTS_TEMP_FINAL = 0.2  # Final MCTS temperature for exploitation
BUFFER_SIZE = 5000000  # Size of replay buffer
BATCH_SIZE = 4096  # Batch size for training
NUM_EPISODES = 3000  # Total training episodes
EVAL_INTERVAL = 50  # Episodes between evaluations
CHECKPOINT_INTERVAL = 100  # Episodes between checkpoints
NUM_WORKERS = min(16, cpu_count() - 1)  # Number of parallel workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for computation
LEARNING_RATE = 0.001  # Initial learning rate
LR_SCHEDULE = [(1000, 0.0005), (2000, 0.0001)]  # Learning rate schedule
USE_AMP = torch.cuda.is_available()  # Enable mixed precision training
USE_DISTRIBUTED = torch.cuda.device_count() > 1  # Enable distributed training

# Configure logging with detailed format
logging.basicConfig(
    filename=f"alphazero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define transition tuple for prioritized experience replay
Transition = namedtuple('Transition', ('state', 'policy', 'value', 'priority', 'game_stage'))

# ======================
# NEURAL NETWORK
# ======================
class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and relative positional encoding."""
    def __init__(self, channels, heads=ATTENTION_HEADS):
        """
        Initialize Transformer block.

        Args:
            channels (int): Number of input/output channels.
            heads (int): Number of attention heads.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, heads, dropout=0.1)
        self.norm1 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels)
        )
        self.norm2 = nn.LayerNorm(channels)
        self.dropout2 = nn.Dropout(0.1)
        self.rel_pos_encoding = nn.Parameter(torch.randn(BOARD_SIZE * BOARD_SIZE, channels) * 0.02)

    def forward(self, x):
        """
        Forward pass of the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)  # Reshape for attention
        pos_enc = self.rel_pos_encoding.unsqueeze(1).expand(-1, b, -1)
        x_with_pos = x_flat + pos_enc
        attn_out, _ = self.attention(x_with_pos, x_with_pos, x_with_pos)
        x_flat = self.norm1(x_flat + self.dropout1(attn_out))
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + self.dropout2(ffn_out))
        return x_flat.permute(1, 2, 0).view(b, c, h, w)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        """
        Initialize SE block.

        Args:
            channels (int): Number of input/output channels.
            reduction (int): Reduction ratio for squeeze operation.
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the SE block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with channel-wise scaling.
        """
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNextBlock(nn.Module):
    """ResNeXt block with grouped convolutions and SE attention."""
    def __init__(self, channels, cardinality=8):
        """
        Initialize ResNeXt block.

        Args:
            channels (int): Number of input/output channels.
            cardinality (int): Number of groups for grouped convolution.
        """
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, padding=1, groups=cardinality))
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        """
        Forward pass of the ResNeXt block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with residual connection.
        """
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        if self.training:
            x = self.dropout(x)
        return torch.relu(x + residual)

class GraphConvBlock(nn.Module):
    """Graph convolution block for modeling board topology."""
    def __init__(self, channels):
        """
        Initialize GCN block.

        Args:
            channels (int): Number of input/output channels.
        """
        super().__init__()
        self.gcn = GCNConv(channels, channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN block.

        Args:
            x (torch.Tensor): Node features of shape (batch * nodes, channels).
            edge_index (torch.Tensor): Graph edge indices of shape (2, edges).

        Returns:
            torch.Tensor: Output node features.
        """
        x = self.gcn(x, edge_index)
        x = self.bn(x)
        return torch.relu(x)

class HybridOthelloNet(nn.Module):
    """Hybrid neural network combining CNN, Transformer, and GNN for Othello policy and value prediction."""
    def __init__(self):
        """Initialize the hybrid network with ResNeXt, Transformer, GNN, and memory components."""
        super().__init__()
        self.conv_init = nn.Conv2d(NUM_CHANNELS, 512, 3, padding=1)
        self.bn_init = nn.BatchNorm2d(512)
        self.res_blocks = nn.ModuleList([ResNextBlock(512) for _ in range(NUM_RES_BLOCKS)])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(512) for _ in range(TRANSFORMER_LAYERS)])
        self.gnn_layers = nn.ModuleList([GraphConvBlock(512) for _ in range(GNN_LAYERS)])
        edge_index = self._create_othello_graph()
        self.register_buffer('edge_index', edge_index)
        self.policy_conv = nn.Conv2d(512, 256, 1)
        self.policy_bn = nn.BatchNorm2d(256)
        self.policy_fc1 = nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, 1024)
        self.policy_fc2 = nn.Linear(1024, BOARD_SIZE * BOARD_SIZE)
        self.aux_policy_fc = nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        self.value_conv = nn.Conv2d(512, 256, 1)
        self.value_bn = nn.BatchNorm2d(256)
        self.value_fc1 = nn.Linear(256 * BOARD_SIZE * BOARD_SIZE, 1024)
        self.value_fc2 = nn.Linear(1024, 256)
        self.value_fc3 = nn.Linear(256, 1)
        self.aux_value_fc = nn.Linear(256, 1)
        self.memory_key = nn.Linear(512, 128)
        self.memory_value = nn.Linear(512, 128)
        self.memory_query = nn.Linear(512, 128)
        self.to(DEVICE)
        if hasattr(torch, 'compile'):
            self.compile(mode="reduce-overhead")  # Optimize with torch.compile if available

    def _create_othello_graph(self):
        """
        Create edge indices for the Othello board graph, connecting adjacent squares.

        Returns:
            torch.Tensor: Edge indices of shape (2, edges).
        """
        edges = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                node_id = i * BOARD_SIZE + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                            neighbor_id = ni * BOARD_SIZE + nj
                            edges.append([node_id, neighbor_id])
        return torch.tensor(edges, dtype=torch.long).t().contiguous().to(DEVICE)

    def forward(self, x):
        """
        Forward pass of the hybrid network, producing policy and value outputs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            tuple: (policy, value) during inference, or (policy, value, aux_policy, aux_value) during training.
        """
        b, c, h, w = x.shape
        x = torch.relu(self.bn_init(self.conv_init(x)))  # Initial convolution
        for res_block in self.res_blocks:
            x = res_block(x)  # ResNeXt backbone
        for transformer in self.transformer_blocks:
            x = transformer(x)  # Transformer processing
        # GNN processing
        x_graph = x.view(b, 512, -1).permute(0, 2, 1).reshape(-1, 512)
        batch_edge_index = self.edge_index.repeat(1, b) + torch.arange(0, b * 64, 64, device=DEVICE).repeat_interleave(self.edge_index.size(1))
        for gnn_layer in self.gnn_layers:
            x_graph = gnn_layer(x_graph, batch_edge_index)
        x_graph = x_graph.view(b, h * w, 512).permute(0, 2, 1).view(b, 512, h, w)
        x = x + 0.2 * x_graph  # Residual connection with GNN features
        # Memory network
        mem_flat = x.view(b, 512, -1).permute(0, 2, 1)
        keys = self.memory_key(mem_flat)
        values = self.memory_value(mem_flat)
        queries = self.memory_query(mem_flat.mean(dim=1, keepdim=True))
        attn_weights = torch.softmax(torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(128), dim=-1)
        mem_output = torch.bmm(attn_weights, values)
        # Policy head
        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p_flat = p.view(b, -1)
        p = torch.relu(self.policy_fc1(p_flat))
        policy_logits = self.policy_fc2(p)
        aux_policy_logits = self.aux_policy_fc(p_flat)
        mem_policy_bias = torch.relu(torch.bmm(mem_output, keys.transpose(1, 2)).view(b, -1))
        policy_logits = policy_logits + 0.1 * mem_policy_bias  # Add memory-based bias
        policy = torch.softmax(policy_logits, dim=1)
        aux_policy = torch.softmax(aux_policy_logits, dim=1)
        # Value head
        v = torch.relu(self.value_bn(self.value_conv(x)))
        v_flat = v.view(b, -1)
        v = torch.relu(self.value_fc1(v_flat))
        v = torch.relu(self.value_fc2(v))
        value = torch.tanh(self.value_fc3(v))
        aux_value = self.aux_value_fc(v)
        if self.training:
            return policy, value, aux_policy, aux_value
        else:
            return policy, value

# ======================
# ENHANCED MCTS
# ======================
class MCTSNode:
    """MCTS node for game tree search with virtual loss and advanced statistics."""
    def __init__(self, game_state, player, parent=None, prior=0.0, move=None):
        """
        Initialize MCTS node.

        Args:
            game_state (Othello): Current game state.
            player (Player): Current player.
            parent (MCTSNode, optional): Parent node.
            prior (float): Prior probability from neural network.
            move (tuple, optional): Move leading to this node.
        """
        self.game = game_state
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.prior = prior
        self.move = move
        self.virtual_loss = 0
        self.mean_value = 0.0
        self.var_value = 0.0
        self.min_value = 1.0
        self.max_value = -1.0
        self.ucb_coeff = 2.0

    def is_leaf(self):
        """Check if node is a leaf (no children)."""
        return not self.children

    def add_virtual_loss(self, amount=3.0):
        """Add virtual loss to discourage over-exploration."""
        self.virtual_loss += amount

    def remove_virtual_loss(self, amount=3.0):
        """Remove virtual loss after backpropagation."""
        self.virtual_loss -= amount

    def update_statistics(self, value):
        """
        Update node statistics with new value.

        Args:
            value (float): Value to update statistics with.
        """
        old_mean = self.mean_value
        old_var = self.var_value
        self.mean_value = old_mean + (value - old_mean) / self.visits
        self.var_value = old_var + (value - old_mean) * (value - self.mean_value)
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.ucb_coeff = 2.0 + 0.5 * (self.var_value / (self.visits + 1e-8))

    def expand(self, policy_probs, valid_moves):
        """
        Expand node by creating children for valid moves.

        Args:
            policy_probs (np.ndarray): Policy probabilities from neural network.
            valid_moves (list): List of valid (row, col) moves.
        """
        total_prior = sum(policy_probs[m[0] * 8 + m[1]] for m in valid_moves)
        for move in valid_moves:
            new_game = copy.deepcopy(self.game)
            new_game.apply_move(move[0], move[1], self.player)
            flat_idx = move[0] * 8 + move[1]
            prior = policy_probs[flat_idx] / (total_prior + 1e-8)
            self.children[move] = MCTSNode(new_game, self.player.opponent(), self, prior, move)

    def select_child(self, noise_eps=0.0, fpu_reduction=0.0):
        """
        Select child node using PUCT formula with enhancements.

        Args:
            noise_eps (float): Noise factor for exploration.
            fpu_reduction (float): First-play urgency reduction factor.

        Returns:
            tuple: (move, child_node) selected.
        """
        total_visits = sum(child.visits for child in self.children.values()) + 1e-8
        sqrt_total = np.sqrt(total_visits)
        best_score = -np.inf
        best_move = None
        best_child = None
        fpu_value = 1.0 - fpu_reduction
        for move, child in self.children.items():
            if child.virtual_loss > 10:
                continue
            q_value = child.total_value / (child.visits + child.virtual_loss + 1e-8)
            if child.visits == 0:
                q_value = fpu_value
            prior = child.prior
            if noise_eps > 0.0 and self.parent is None:
                prior = (1 - noise_eps) * prior + noise_eps * self.noise.get(move, 0.0)
            explore = prior * sqrt_total / (child.visits + child.virtual_loss + 1.0)
            score = q_value + child.ucb_coeff * explore
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

class EnhancedAlphaZeroAgent:
    """Advanced AlphaZero agent with MCTS and hybrid neural network."""
    def __init__(self, player, rank=0):
        """
        Initialize the AlphaZero agent.

        Args:
            player (Player): Player (Black or White).
            rank (int): Rank for distributed training.
        """
        self.net = HybridOthelloNet()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        milestones = [milestone for milestone, _ in LR_SCHEDULE]
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.player = player
        self.scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
        self.best_loss = float('inf')
        self.patience = 0
        self.max_patience = 10
        self.temp_schedule = np.linspace(MCTS_TEMP_INIT, MCTS_TEMP_FINAL, NUM_EPISODES)
        if USE_DISTRIBUTED:
            self.net = DDP(self.net, device_ids=[rank])
        self.rank = rank

    def state_to_tensor(self, game, move_history):
        """
        Convert game state to neural network input tensor.

        Args:
            game (Othello): Current game state.
            move_history (list): List of past moves.

        Returns:
            torch.Tensor: Input tensor of shape (1, channels, height, width).
        """
        board = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        current_player = 1 if game.current_player == self.player else 0
        board[0].fill(current_player)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == self.player.value:
                    board[1][r][c] = 1.0
                elif game.board[r][c] != ".":
                    board[2][r][c] = 1.0
        for i, move in enumerate(move_history[-8:], 3):
            if move:
                board[i][move[0]][move[1]] = 1.0
        valid_moves = game.get_valid_moves(game.current_player)
        for move in valid_moves:
            board[11][move[0]][move[1]] = 1.0
        game.switch_player()
        opponent_moves = game.get_valid_moves(game.current_player)
        for move in opponent_moves:
            board[12][move[0]][move[1]] = 1.0
        game.switch_player()
        return torch.tensor(board).unsqueeze(0).to(DEVICE)

    def mcts_search(self, game, move_history, simulations=MCTS_SIMULATIONS, temp=1.0):
        """
        Perform MCTS search to generate policy and value estimates.

        Args:
            game (Othello): Current game state.
            move_history (list): List of past moves.
            simulations (int): Number of MCTS simulations.
            temp (float): Temperature for policy softening.

        Returns:
            tuple: (policy, value) where policy is a probability distribution and value is the estimated outcome.
        """
        root = MCTSNode(game, self.player)
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return np.zeros(64), 0.0
        with torch.no_grad():
            state_tensor = self.state_to_tensor(game, move_history)
            policy, value = self.net(state_tensor)
            policy_probs = policy.squeeze().cpu().numpy()
        if valid_moves:
            filled = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] != ".")
            game_stage = filled / (BOARD_SIZE * BOARD_SIZE)
            dirichlet_alpha = 0.3 if game_stage < 0.5 else 0.15  # Adjust noise based on game stage
            noise = np.random.dirichlet([dirichlet_alpha] * len(valid_moves))
            root.noise = {}
            for i, move in enumerate(valid_moves):
                idx = move[0] * 8 + move[1]
                root.noise[move] = noise[i]
            root.expand(policy_probs, valid_moves)
        for simulation in range(simulations):
            node = root
            search_path = [node]
            while not node.is_leaf():
                fpu_reduction = 0.1 * min(1.0, simulation / (simulations * 0.5))
                noise_eps = 0.25 if simulation < simulations * 0.25 else 0.0
                move, node = node.select_child(noise_eps, fpu_reduction)
                node.add_virtual_loss()
                search_path.append(node)
            if not node.game.is_game_over() and node.visits > 0:
                with torch.no_grad():
                    state_tensor = self.state_to_tensor(node.game, move_history)
                    policy, value = self.net(state_tensor)
                    policy_probs = policy.squeeze().cpu().numpy()
                    value = value.item()
                valid_moves = node.game.get_valid_moves(node.player)
                if valid_moves:
                    node.expand(policy_probs, valid_moves)
            elif node.game.is_game_over():
                x, o = node.game.count_discs()
                if x > o:
                    value = 1 if self.player == Player.BLACK else -1
                elif o > x:
                    value = -1 if self.player == Player.BLACK else 1
                else:
                    value = 0
            else:
                with torch.no_grad():
                    state_tensor = self.state_to_tensor(node.game, move_history)
                    _, value = self.net(state_tensor)
                    value = value.item()
            for node in reversed(search_path):
                node.visits += 1
                node.total_value += value
                node.update_statistics(value)
                node.remove_virtual_loss()
                value = -value
        visit_counts = np.zeros(64)
        for move, child in root.children.items():
            idx = move[0] * 8 + move[1]
            visit_counts[idx] = child.visits
        if temp > 0:
            visit_counts = visit_counts ** (1 / temp)  # Apply temperature
        normalized_policy = visit_counts / (visit_counts.sum() + 1e-8)
        root_value = root.total_value / (root.visits + 1e-8)
        return normalized_policy, root_value

    def train(self, batch, game_stage=None):
        """
        Train the neural network on a batch of transitions.

        Args:
            batch (list): List of Transition tuples.
            game_stage (float, optional): Game stage for curriculum learning.

        Returns:
            float: Total loss value.
        """
        states, policies, values, _, _ = zip(*batch)
        states = torch.cat(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(DEVICE)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(DEVICE)
        self.optimizer.zero_grad()
        if self.scaler:
            with torch.cuda.amp.autocast():
                pred_policies, pred_values, aux_policies, aux_values = self.net(states)
                smooth_policies = 0.9 * policies + 0.1 / 64  # Label smoothing
                policy_loss = -(smooth_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
                aux_policy_loss = -(smooth_policies * torch.log(aux_policies + 1e-8)).sum(dim=1).mean()
                value_loss = nn.HuberLoss(delta=0.5)(pred_values.squeeze(), values)
                move_count = torch.ones_like(values) * 10  # Placeholder for move count prediction
                aux_value_loss = nn.MSELoss()(aux_values.squeeze(), move_count)
                loss = policy_loss + value_loss + 0.5 * aux_policy_loss + 0.3 * aux_value_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred_policies, pred_values, aux_policies, aux_values = self.net(states)
            smooth_policies = 0.9 * policies + 0.1 / 64
            policy_loss = -(smooth_policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
            aux_policy_loss = -(smooth_policies * torch.log(aux_policies + 1e-8)).sum(dim=1).mean()
            value_loss = nn.HuberLoss(delta=0.5)(pred_values.squeeze(), values)
            move_count = torch.ones_like(values) * 10
            aux_value_loss = nn.MSELoss()(aux_values.squeeze(), move_count)
            loss = policy_loss + value_loss + 0.5 * aux_policy_loss + 0.3 * aux_value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
        self.scheduler.step()
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.patience = 0
        else:
            self.patience += 1
        wandb.log({"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()})
        return loss.item()

    def save_checkpoint(self, path, episode):
        """
        Save model checkpoint.

        Args:
            path (str): File path for checkpoint.
            episode (int): Current episode number.
        """
        state = {
            'state_dict': self.net.module.state_dict() if USE_DISTRIBUTED else self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'episode': episode,
            'best_loss': self.best_loss,
            'patience': self.patience
        }
        torch.save(state, path)
        logging.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """
        Load model checkpoint.

        Args:
            path (str): File path of checkpoint.

        Returns:
            int: Episode number from checkpoint, or 0 if not found.
        """
        if os.path.exists(path):
            state = torch.load(path, map_location=DEVICE)
            if USE_DISTRIBUTED:
                self.net.module.load_state_dict(state['state_dict'])
            else:
                self.net.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.best_loss = state.get('best_loss', float('inf'))
            self.patience = state.get('patience', 0)
            logging.info(f"Checkpoint loaded: {path}")
            return state.get('episode', 0)
        return 0

# ======================
# TRAINING FRAMEWORK
# ======================
class ReplayDataset(Dataset):
    """Dataset for prioritized experience replay buffer."""
    def __init__(self, buffer):
        """
        Initialize dataset with replay buffer.

        Args:
            buffer (deque): Replay buffer containing transitions.
        """
        self.buffer = buffer

    def __len__(self):
        """Return size of replay buffer."""
        return len(self.buffer)

    def __getitem__(self, idx):
        """
        Get transition at index.

        Args:
            idx (int): Index of transition.

        Returns:
            Transition: Transition tuple.
        """
        return self.buffer[idx]

class AdvancedRLTrainingManager:
    """Training manager for AlphaZero self-play and neural network optimization."""
    def __init__(self, rank=0, world_size=1):
        """
        Initialize training manager.

        Args:
            rank (int): Rank for distributed training.
            world_size (int): Number of processes in distributed training.
        """
        self.rank = rank
        self.world_size = world_size
        self.agent = EnhancedAlphaZeroAgent(Player.BLACK, rank)
        self.opponents = [
            AdvancedOthelloAI(Player.WHITE, 3),
            AdvancedOthelloAI(Player.WHITE, 5),
            AdvancedOthelloAI(Player.WHITE, 8)
        ]
        self.best_win_rate = -np.inf
        self.start_time = time.time()
        if rank == 0:
            wandb.init(project="advanced-alphazero", config={
                "board_size": BOARD_SIZE,
                "num_channels": NUM_CHANNELS,
                "num_res_blocks": NUM_RES_BLOCKS,
                "mcts_simulations": MCTS_SIMULATIONS,
                "num_episodes": NUM_EPISODES
            })

    def self_play_game(self, episode):
        """
        Run a single self-play game to generate training data.

        Args:
            episode (int): Current episode number.

        Returns:
            list: List of Transition tuples from the game.
        """
        game = Othello()
        history = []
        move_history = []
        simulations = MCTS_SIMULATIONS if episode < NUM_EPISODES // 2 else MCTS_SIMULATIONS * 2
        temp = self.agent.temp_schedule[episode]
        move_count = 0
        while not game.is_game_over():
            valid_moves = game.get_valid_moves(game.current_player)
            if not valid_moves:
                game.switch_player()
                continue
            filled = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if game.board[r][c] != ".")
            game_stage = filled / (BOARD_SIZE * BOARD_SIZE)
            policy, value = self.agent.mcts_search(game, move_history, simulations, temp)
            full_policy = np.zeros(64, dtype=np.float32)
            valid_indices = [m[0] * 8 + m[1] for m in valid_moves]
            total = sum(policy[i] for i in valid_indices)
            full_policy[valid_indices] = policy[valid_indices] / (total + 1e-8)
            chosen_idx = np.random.choice(len(valid_moves), p=full_policy[valid_indices])
            chosen_move = valid_moves[chosen_idx]
            game.apply_move(chosen_move[0], chosen_move[1], game.current_player)
            move_history.append(chosen_move)
            priority = abs(value) + 0.1 * (1 - game_stage)
            history.append(Transition(
                self.agent.state_to_tensor(game, move_history),
                full_policy,
                None,
                priority,
                game_stage
            ))
            game.switch_player()
            move_count += 1
        x, o = game.count_discs()
        final_value = 1 if (x > o and self.agent.player == Player.BLACK) else -1
        for i in range(len(history)):
            history[i] = history[i]._replace(value=final_value * (-1) ** i)
        logging.info(f"Self-play game completed: Moves={move_count}, Score=X:{x} O:{o}")
        return history

    def evaluate(self, num_games=50, opponent_idx=1):
        """
        Evaluate agent against a traditional AI opponent.

        Args:
            num_games (int): Number of evaluation games.
            opponent_idx (int): Index of opponent AI (0: depth 3, 1: depth 5, 2: depth 8).

        Returns:
            float: Win rate percentage.
        """
        wins = 0
        opponent = self.opponents[opponent_idx]
        for _ in range(num_games):
            game = Othello()
            move_history = []
            while not game.is_game_over():
                if game.current_player == self.agent.player:
                    policy, _ = self.agent.mcts_search(game, move_history, MCTS_SIMULATIONS // 2, temp=0.1)
                    valid_moves = game.get_valid_moves(game.current_player)
                    move_idx = np.argmax([policy[m[0] * 8 + m[1]] for m in valid_moves])
                    move = valid_moves[move_idx]
                else:
                    move = opponent.find_best_move(game)
                game.apply_move(move[0], move[1], game.current_player)
                move_history.append(move)
                game.switch_player()
            x, o = game.count_discs()
            if (x > o and self.agent.player == Player.BLACK) or (o > x and self.agent.player == Player.WHITE):
                wins += 1
        win_rate = (wins / num_games) * 100
        logging.info(f"Evaluation | Opponent Depth: {opponent.search_depth} | Win Rate: {win_rate:.1f}%")
        if self.rank == 0:
            wandb.log({"win_rate": win_rate, "opponent_depth": opponent.search_depth})
        return win_rate

    def tournament(self, num_games=20):
        """
        Run a tournament to select the best model checkpoint.

        Args:
            num_games (int): Number of games per checkpoint evaluation.
        """
        checkpoints = [f"checkpoint_ep{i}.pth" for i in range(CHECKPOINT_INTERVAL, NUM_EPISODES + 1, CHECKPOINT_INTERVAL)]
        best_checkpoint = None
        best_win_rate = -np.inf
        for checkpoint in checkpoints:
            if os.path.exists(checkpoint):
                self.agent.load_checkpoint(checkpoint)
                win_rate = self.evaluate(num_games, opponent_idx=1)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_checkpoint = checkpoint
        if best_checkpoint:
            self.agent.load_checkpoint(best_checkpoint)
            logging.info(f"Tournament | Best Checkpoint: {best_checkpoint} | Win Rate: {best_win_rate:.1f}%")
            if self.rank == 0:
                wandb.log({"tournament_win_rate": best_win_rate})

    def train_cycle(self, num_episodes=NUM_EPISODES):
        """
        Run the full training cycle with self-play, training, and evaluation.

        Args:
            num_episodes (int): Total number of episodes to train.

        Raises:
            Exception: If training fails, logs error and re-raises.
        """
        logging.info(f"Training Start | Episodes: {num_episodes} | Workers: {NUM_WORKERS} | Rank: {self.rank}")
        if self.rank == 0:
            start_episode = self.agent.load_checkpoint("best_model.pth")
        else:
            start_episode = 0
        if USE_DISTRIBUTED:
            dist.barrier()  # Synchronize processes
        try:
            with Pool(NUM_WORKERS) as pool:
                for episode in range(start_episode, num_episodes):
                    start_time = time.time()
                    results = pool.starmap(self.self_play_game, [(episode + i,) for i in range(NUM_WORKERS)])
                    for game_data in results:
                        self.agent.replay_buffer.extend(game_data)
                    if len(self.agent.replay_buffer) >= BATCH_SIZE:
                        dataset = ReplayDataset(self.agent.replay_buffer)
                        priorities = np.array([t.priority for t in self.agent.replay_buffer])
                        probs = priorities / priorities.sum()
                        indices = np.random.choice(len(self.agent.replay_buffer), BATCH_SIZE, p=probs)
                        batch = [self.agent.replay_buffer[i] for i in indices]
                        loss = self.agent.train(batch)
                        logging.info(f"Episode {episode + 1} | Loss: {loss:.4f}")
                    if (episode + 1) % EVAL_INTERVAL == 0 and self.rank == 0:
                        win_rate = self.evaluate()
                        if win_rate > self.best_win_rate:
                            self.best_win_rate = win_rate
                            self.agent.save_checkpoint("best_model.pth", episode + 1)
                    if (episode + 1) % CHECKPOINT_INTERVAL == 0 and self.rank == 0:
                        self.agent.save_checkpoint(f"checkpoint_ep{episode + 1}.pth", episode + 1)
                        self.tournament()
                    if self.agent.patience >= self.agent.max_patience and self.rank == 0:
                        logging.info(f"Early stopping triggered at episode {episode + 1}")
                        break
                    logging.info(f"Episode {episode + 1} | Time: {time.time() - start_time:.1f}s")
                    if self.rank == 0:
                        wandb.log({"episode": episode + 1, "buffer_size": len(self.agent.replay_buffer)})
        except Exception as e:
            logging.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
            raise
        finally:
            if self.rank == 0:
                final_win_rate = self.evaluate(num_games=100)
                wandb.log({"final_win_rate": final_win_rate})
                wandb.finish()

def setup_distributed(rank, world_size):
    """
    Initialize distributed training environment.

    Args:
        rank (int): Rank of the process.
        world_size (int): Total number of processes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main():
    """Main function to start training, handling distributed or single-process setup."""
    world_size = torch.cuda.device_count() if USE_DISTRIBUTED else 1
    if USE_DISTRIBUTED:
        mp.spawn(
            run_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        trainer = AdvancedRLTrainingManager()
        trainer.train_cycle()

def run_worker(rank, world_size):
    """
    Worker function for distributed training.

    Args:
        rank (int): Rank of the process.
        world_size (int): Total number of processes.
    """
    setup_distributed(rank, world_size)
    trainer = AdvancedRLTrainingManager(rank, world_size)
    trainer.train_cycle()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()