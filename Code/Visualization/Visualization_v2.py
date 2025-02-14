import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import re
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

class OthelloVisualizer:
    def __init__(self, game_log):
        self.game_log = game_log
        self.boards = self.extract_boards()
        self.moves = self.extract_moves()
        self.scores = self.extract_scores()
        self.stats = self.extract_ai_stats()
        
        # Custom color scheme
        self.board_color = '#2D5A27'  # Dark green
        self.grid_color = '#1a1a1a'
        self.text_color = '#FFFFFF'
        self.highlight_color = '#FFD700'  # Gold
        
        # Create custom colormap for score progression
        self.score_colors = ['#FF6B6B', '#4ECDC4']  # Red to Teal
        
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
        
        # Set up the figure with dark theme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 9), facecolor='#2b2b2b')
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
        
        # Board subplot
        ax1 = fig.add_subplot(gs[:, 0], facecolor=self.board_color)
        # Score subplot
        ax2 = fig.add_subplot(gs[0, 1], facecolor='#1f1f1f')
        # Stats subplot
        ax3 = fig.add_subplot(gs[1, 1], facecolor='#1f1f1f')
        
        def update(frame):
            # Clear all axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            # Draw board
            board = self.boards[frame]
            
            # Set up the grid
            ax1.set_xlim(-0.5, 7.5)
            ax1.set_ylim(-0.5, 7.5)
            
            # Create board background
            board_bg = Rectangle((-0.5, -0.5), 8, 8, facecolor=self.board_color)
            ax1.add_patch(board_bg)
            
            # Add grid lines
            for i in range(8):
                for j in range(8):
                    square = Rectangle((j-0.5, i-0.5), 1, 1, 
                                    facecolor=self.board_color,
                                    edgecolor=self.grid_color,
                                    linewidth=1)
                    ax1.add_patch(square)
            
            # Draw pieces with effects
            for i in range(8):
                for j in range(8):
                    if board[i][j] in ['X', 'O']:
                        # Add shadow
                        shadow = Circle((j+0.05, 7-i-0.05), 0.4, 
                                     color=(0, 0, 0, 0.3),
                                     zorder=2)
                        ax1.add_patch(shadow)
                        
                        # Add piece
                        color = 'black' if board[i][j] == 'X' else 'white'
                        piece = Circle((j, 7-i), 0.4, 
                                    color=color,
                                    ec='#333333',
                                    linewidth=2,
                                    zorder=3)
                        ax1.add_patch(piece)
                        
                        # Add highlight
                        if color == 'black':
                            highlight = Circle((j-0.1, 7-i+0.1), 0.1,
                                            color=(1, 1, 1, 0.3),
                                            zorder=4)
                            ax1.add_patch(highlight)
            
            # Highlight last move if available
            if frame > 0 and frame <= len(self.moves):
                last_move = self.moves[frame-1]
                if len(last_move[1]) == 2:  # Valid move format (e.g., 'd3')
                    col = ord(last_move[1][0].lower()) - ord('a')
                    row = 8 - int(last_move[1][1])
                    highlight = Rectangle((col-0.5, row-0.5), 1, 1,
                                       facecolor='none',
                                       edgecolor=self.highlight_color,
                                       linewidth=2,
                                       linestyle='--',
                                       zorder=1)
                    ax1.add_patch(highlight)
            
            # Set up coordinates
            x_ticks = np.arange(8)
            y_ticks = np.arange(8)
            ax1.set_xticks(x_ticks)
            ax1.set_yticks(y_ticks)
            ax1.set_xticklabels(list('abcdefgh'))
            ax1.set_yticklabels(list('87654321'))
            
            # Style the coordinate labels
            ax1.tick_params(colors=self.text_color, labelsize=12)
            
            # Add move counter with fancy styling
            title = ax1.set_title(f'Move {frame + 1}/{len(self.boards)}',
                                color=self.text_color,
                                fontsize=14,
                                pad=20)
            title.set_path_effects([path_effects.withStroke(linewidth=3, 
                                                          foreground='#333333')])
            
            # Draw scores with gradient fill
            if frame < len(self.scores['X']):
                moves = range(frame + 1)
                scores_x = self.scores['X'][:frame+1]
                scores_o = self.scores['O'][:frame+1]
                
                # Plot lines with gradient fill
                ax2.fill_between(moves, scores_x, alpha=0.3, color=self.score_colors[0])
                ax2.fill_between(moves, scores_o, alpha=0.3, color=self.score_colors[1])
                ax2.plot(moves, scores_x, color=self.score_colors[0], 
                        label='Black (X)', linewidth=2)
                ax2.plot(moves, scores_o, color=self.score_colors[1], 
                        label='White (O)', linewidth=2)
                
                ax2.set_title('Score Progression', color=self.text_color, fontsize=12)
                ax2.set_xlabel('Move', color=self.text_color)
                ax2.set_ylabel('Score', color=self.text_color)
                legend = ax2.legend(facecolor='#333333', edgecolor='none')
                plt.setp(legend.get_texts(), color=self.text_color)
                ax2.grid(True, alpha=0.2)
                ax2.tick_params(colors=self.text_color)
            
            # Draw AI stats with enhanced styling
            if frame < len(self.stats):
                stat = self.stats[frame]
                bars = ax3.bar(['Nodes', 'Time (ms)'], 
                             [stat['nodes'], stat['time']*1000],
                             color=self.score_colors)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height):,}',
                            ha='center', va='bottom', color=self.text_color)
                
                ax3.set_title('AI Performance', color=self.text_color, fontsize=12)
                ax3.set_yscale('log')
                ax3.tick_params(colors=self.text_color)
                ax3.grid(True, alpha=0.2)
            
            plt.tight_layout()
        
        # Create animation with smoother frame rate
        anim = FuncAnimation(fig, update, frames=len(self.boards),
                           interval=750, repeat=True)
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
                   xticklabels=list('abcdefgh'), yticklabels=list('12345678'))
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
        anim.save('othello_game_v2.gif', writer='pillow')
        print("Displaying animation...")
        print("Animation saved as 'othello_game_v2.gif'")
        
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
        # Find all .txt files in the current directory
        txt_files = glob.glob("*.txt")
        
        # Filter out 'requirements.txt' if it exists
        txt_files = [file for file in txt_files if os.path.basename(file) != "requirements.txt"]
        print(f"Found {len(txt_files)} .txt files in the current directory.")
        
        # Check if there are any .txt files left after filtering
        if not txt_files:
            print("Error: No valid .txt files found in the current directory (excluding 'requirements.txt').")
        else:
            # Use the first .txt file found (after filtering)
            file_name = txt_files[0]
            print(f"Reading game log file: {file_name}...")
            
            # Open and read the file
            with open(file_name, 'r') as f:
                game_log = f.read()
            
            # Print some information about the game log
            print("Game log length:", len(game_log))
            print("First few lines of game log:")
            print('\n'.join(game_log.split('\n')[:10]))
            
            # Visualize the game log
            visualize_game(game_log)
    except Exception as e:
        # Handle any unexpected errors and print the traceback
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()