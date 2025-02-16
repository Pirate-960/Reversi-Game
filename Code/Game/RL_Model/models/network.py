import torch
import torch.nn as nn
from config import config

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)

class OthelloNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(config.num_channels, config.hidden_channels, 3, padding=1),
            nn.BatchNorm2d(config.hidden_channels),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResBlock(config.hidden_channels) 
            for _ in range(config.num_res_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.hidden_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * config.board_size * config.board_size, 
                     config.action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(config.hidden_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * config.board_size * config.board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv_input(x)
        for res_block in self.res_blocks:
            x = res_block(x)
            
        policy = self.policy_head(x)
        policy = torch.softmax(policy, dim=1)
        
        value = self.value_head(x)
        
        return policy, value