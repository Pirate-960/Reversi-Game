import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from models.network import OthelloNet
from training.self_play import SelfPlayWorker
from config import config
import os

class TrainingManager:
    def __init__(self):
        self.network = OthelloNet().to(config.device)
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scaler = GradScaler()  # For mixed precision training
        self.worker = SelfPlayWorker(self.network)
        
    def train(self):
        """Main training loop"""
        episode = 0
        temperature = 1.0
        
        while episode < config.num_episodes:
            # Self-play phase
            self.network.eval()
            winner = self.worker.execute_episode(temperature)
            
            # Training phase
            batch = self.worker.get_batch(config.batch_size)
            if batch is not None:
                self.network.train()
                loss = self.train_on_batch(batch)
                
                # Log metrics
                wandb.log({
                    "episode": episode,
                    "loss": loss,
                    "temperature": temperature,
                    "buffer_size": len(self.worker.game_buffer)
                })
                
                # Save checkpoint
                if episode % 100 == 0:
                    self.save_checkpoint(f"checkpoint_{episode}.pt")
                
                episode += 1
                
                # Decay temperature
                if episode > 0 and episode % 100 == 0:
                    temperature = max(0.1, temperature * 0.95)
    
    def train_on_batch(self, batch):
        """Train on a single batch"""
        states, policies, values = batch
        
        self.optimizer.zero_grad()
        
        with autocast():
            # Forward pass
            pred_policies, pred_values = self.network(states)
            
            # Calculate losses
            policy_loss = -(policies * torch.log(pred_policies + 1e-8)).sum(dim=1).mean()
            value_loss = nn.MSELoss()(pred_values.squeeze(), values)
            
            total_loss = (config.policy_weight * policy_loss + 
                         config.value_weight * value_loss)
        
        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss.item()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        path = os.path.join(config.checkpoint_dir, filename)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'buffer': self.worker.game_buffer
        }, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = os.path.join(config.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.worker.game_buffer = checkpoint['buffer']