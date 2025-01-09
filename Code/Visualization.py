import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import re
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle

class OthelloVisualizer:
    def __init__(self, game_log):
        self.game_log = game_log
        self.boards = self.extract_boards()
        self.moves = self.extract_moves()
        self.scores = self.extract_scores()
        self.stats = self.extract_ai_stats()
        
    def extract_boards(self):
        boards = []
        current_board = []
        for line in self.game_log.split('\n'):
            if re.match(r'^[1-8]', line):
                row = line[2:].strip()
                current_board.append([c for c in row.split()])
            elif current_board and line.startswith('  a b'):
                boards.append(np.array(current_board))
                current_board = []
        if current_board:  # Append the last board if exists
            boards.append(np.array(current_board))
        return boards
    
    def extract_moves(self):
        moves = []
        for line in self.game_log.split('\n'):
            if "plays:" in line:
                player = 'X' if 'X plays' in line else 'O'
                move = line.split("plays:")[-1].strip()
                moves.append((player, move))
        return moves
    
    def extract_scores(self):
        scores = {'X': [], 'O': []}
        for line in self.game_log.split('\n'):
            if "Current score" in line:
                match = re.search(r'Black \(X\): (\d+), White \(O\): (\d+)', line)
                if match:
                    scores['X'].append(int(match.group(1)))
                    scores['O'].append(int(match.group(2)))
        # Ensure we have at least one score
        if not scores['X']:
            scores['X'].append(2)
            scores['O'].append(2)
        return scores
    
    def extract_ai_stats(self):
        stats = []
        lines = self.game_log.split('\n')
        i = 0
        while i < len(lines):
            if "AI Statistics:" in lines[i]:
                try:
                    stats_dict = {'nodes': 0, 'time': 0.0}
                    # Look at next few lines for stats
                    for j in range(i, min(i + 4, len(lines))):
                        nodes_match = re.search(r'Nodes evaluated: (\d+)', lines[j])
                        time_match = re.search(r'Time taken: ([\d.]+)', lines[j])
                        if nodes_match:
                            stats_dict['nodes'] = int(nodes_match.group(1))
                        if time_match:
                            stats_dict['time'] = float(time_match.group(1))
                    stats.append(stats_dict)
                except Exception as e:
                    print(f"Warning: Error processing AI stats at line {i}: {e}")
                    stats.append({'nodes': 0, 'time': 0.0})
            i += 1
        return stats
    
    def animate_game(self):
        if not self.boards:
            raise ValueError("No board states found in the game log")
            
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1])
        
        # Board subplot
        ax1 = fig.add_subplot(gs[:, 0])
        # Score subplot
        ax2 = fig.add_subplot(gs[0, 1])
        # Stats subplot
        ax3 = fig.add_subplot(gs[1, 1])
        
        def update(frame):
            # Clear all axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Draw board
            board = self.boards[frame]
            
            # Set up the grid properly
            ax1.set_xlim(-0.5, 7.5)
            ax1.set_ylim(-0.5, 7.5)
            
            # Create correct number of ticks
            x_ticks = np.arange(8)
            y_ticks = np.arange(8)
            
            ax1.set_xticks(x_ticks)
            ax1.set_yticks(y_ticks)
            
            # Set labels
            ax1.set_xticklabels(list('abcdefgh'))
            ax1.set_yticklabels(list('87654321'))
            
            # Draw grid
            ax1.grid(True)
            
            # Draw pieces
            for i in range(8):
                for j in range(8):
                    if board[i][j] in ['X', 'O']:
                        color = 'black' if board[i][j] == 'X' else 'white'
                        circle = Circle((j, 7-i), 0.4, color=color, ec='black')
                        ax1.add_patch(circle)
            
            ax1.set_title(f'Move {frame + 1}/{len(self.boards)}')
            
            # Draw scores
            if frame < len(self.scores['X']):
                moves = range(frame + 1)
                ax2.plot(moves, self.scores['X'][:frame+1], 'k-', label='Black (X)')
                ax2.plot(moves, self.scores['O'][:frame+1], 'b-', label='White (O)')
                ax2.set_title('Score Progression')
                ax2.set_xlabel('Move')
                ax2.set_ylabel('Score')
                ax2.legend()
                ax2.grid(True)
            
            # Draw AI stats
            if frame < len(self.stats):
                stat = self.stats[frame]
                ax3.bar(['Nodes', 'Time (ms)'], 
                       [stat['nodes'], stat['time']*1000],
                       color=['lightblue', 'lightgreen'])
                ax3.set_title('AI Performance')
                ax3.set_yscale('log')
                
            plt.tight_layout()
            
        anim = FuncAnimation(fig, update, frames=len(self.boards), 
                           interval=1000, repeat=False)
        return anim

    def plot_heatmap(self):
        """Create a heatmap showing frequency of moves on each square"""
        move_freq = np.zeros((8, 8))
        for player, move in self.moves:
            if len(move) == 2:  # Valid move format (e.g., 'd3')
                try:
                    col = ord(move[0].lower()) - ord('a')
                    row = 8 - int(move[1])
                    if 0 <= row < 8 and 0 <= col < 8:
                        move_freq[row][col] += 1
                except (ValueError, IndexError):
                    continue
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(move_freq, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=list('abcdefgh'), yticklabels=list('87654321'))
        plt.title('Move Frequency Heatmap')
        plt.show()
    
    def plot_performance_metrics(self):
        """Plot AI performance metrics over the game"""
        if not self.stats:
            print("No AI statistics available to plot")
            return
            
        data = pd.DataFrame(self.stats)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Nodes evaluated over time
        ax1.plot(data['nodes'], 'b-', marker='o')
        ax1.set_title('Nodes Evaluated per Move')
        ax1.set_ylabel('Nodes')
        ax1.grid(True)
        
        # Processing time over time
        ax2.plot(data['time'] * 1000, 'g-', marker='o')
        ax2.set_title('Processing Time per Move')
        ax2.set_ylabel('Time (ms)')
        ax2.set_xlabel('Move Number')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def visualize_game(game_log):
    try:
        print("Initializing visualizer...")
        visualizer = OthelloVisualizer(game_log)
        
        print(f"Number of boards extracted: {len(visualizer.boards)}")
        print(f"Number of moves extracted: {len(visualizer.moves)}")
        print(f"Number of scores extracted: {len(visualizer.scores['X'])}")
        print(f"Number of AI stats extracted: {len(visualizer.stats)}")
        
        if len(visualizer.boards) == 0:
            print("No board states were extracted from the game log")
            print("First few lines of game log:")
            print('\n'.join(game_log.split('\n')[:10]))
            return
            
        # Create and save the animation
        print("Creating game animation...")
        anim = visualizer.animate_game()
        anim.save('othello_game.gif', writer='pillow')
        print("Displaying animation...")
        print("Animation saved as 'othello_game.gif'")
        
        # Generate additional visualizations
        print("Generating heatmap...")
        visualizer.plot_heatmap()
        print("Heatmap complete!")
        
        print("Generating performance metrics...")
        visualizer.plot_performance_metrics()
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

# If running directly
if __name__ == "__main__":
    try:
        # Read the game log from file
        print("Reading game log file...")
        with open('othello_AIvsAI_d5_h3h3_20250109_215427.txt', 'r') as f:
            game_log = f.read()
        
        print("Game log length:", len(game_log))
        print("First few lines of game log:")
        print('\n'.join(game_log.split('\n')[:10]))
        
        visualize_game(game_log)
    except FileNotFoundError:
        print("Error: Game log file not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()