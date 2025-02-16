import wandb
import torch
from config import config
from training.trainer import TrainingManager

def main():
    # Login check
    try:
        # Login to WANDB
        wandb.login()
        print("✅ Successfully logged into Weights & Biases.")
    except Exception as e:
        print(f"❌ WANDB Login Failed: {e}")
        return

    # Project initialization
    try:
        # Initialize WANDB project
        wandb.init(
            project="alphazero-othello",
            config={
                "num_res_blocks": config.num_res_blocks,
                "num_simulations": config.num_simulations,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate
            }
        )
        print("✅ WANDB Initialized Successfully.")
    except Exception as e:
        print(f"❌ WANDB Initialization Failed: {e}")
        return

    # Initialize training manager
    trainer = TrainingManager()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        trainer.save_checkpoint("interrupted.pt")
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
