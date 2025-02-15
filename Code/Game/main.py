from config import config
from Code.Game.OthelloRLAgent_2 import EnhancedTrainingManager

def main():
    # Initialize training
    manager = EnhancedTrainingManager()
    
    # Start training
    try:
        manager.train_cycle()
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        manager.save_checkpoint("interrupted_checkpoint.pt")

if __name__ == "__main__":
    main()