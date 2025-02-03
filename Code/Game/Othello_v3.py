"""
Othello (Reversi) Game Implementation
This program implements the classic board game Othello with multiple play modes:
- Human vs Human
- Human vs AI
- AI vs AI

The game includes features like:
- Move validation and game state tracking
- Multiple AI heuristics
- Game logging
- Terminal recording
- Various utility functions for game management
"""

import copy
import time
import json
import sys
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import threading
import subprocess
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache

class Player(Enum):
    """Enum for player representation."""
    BLACK = "X"  # Black player
    WHITE = "O"  # White player

    def opponent(self):
        """Return the opponent player."""
        return Player.WHITE if self == Player.BLACK else Player.BLACK

@dataclass
class Move:
    """Dataclass to represent a move."""
    row: int
    col: int
    score: float = 0

class Tee:
    """
    Utility class to enable simultaneous output to console and file.
    Acts as a write multiplexer for output streams.
    
    Attributes:
        files: Tuple of file objects to write to
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        """Write the given object to all registered file streams."""
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure immediate write to avoid buffering

    def flush(self):
        """Flush all file streams."""
        for f in self.files:
            f.flush()


class Othello:
    """
    Main Othello game class implementing game rules and mechanics.
    
    Attributes:
        board (list): 8x8 game board represented as 2D list
        current_player (Player): Current player's symbol (BLACK or WHITE)
        move_history (list): List of all moves made during the game
        game_log (dict): Detailed game state and move information
    """
    def __init__(self):
        """Initialize game board with starting position and game tracking."""
        # Create empty 8x8 board
        self.board = [["." for _ in range(8)] for _ in range(8)]
        # Set up initial four pieces in center
        self.board[3][3], self.board[4][4] = Player.WHITE.value, Player.WHITE.value  # White pieces
        self.board[3][4], self.board[4][3] = Player.BLACK.value, Player.BLACK.value  # Black pieces
        self.current_player = Player.BLACK  # Black plays first
        self.move_history = []
        # Initialize game log with metadata
        self.game_log = {
            'game_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'moves': [],
            'initial_state': self.get_board_state()
        }

    def get_board_state(self):
        """
        Capture current board state including piece positions.
        
        Returns:
            dict: Current board state with piece positions in algebraic notation
        """
        state = {
            'board': [row[:] for row in self.board],
            'disc_positions': {
                Player.BLACK.value: [],
                Player.WHITE.value: []
            }
        }
        # Record positions of all pieces
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == Player.BLACK.value:
                    state['disc_positions'][Player.BLACK.value].append(self.numeric_to_algebraic(i, j))
                elif self.board[i][j] == Player.WHITE.value:
                    state['disc_positions'][Player.WHITE.value].append(self.numeric_to_algebraic(i, j))
        return state

    def algebraic_to_numeric(self, move):
        """
        Convert algebraic notation (e.g., 'e3') to board coordinates.
        
        Args:
            move (str): Move in algebraic notation (e.g., 'e3')
            
        Returns:
            tuple: (row, col) coordinates or None if invalid
        """
        try:
            col = ord(move[0].lower()) - ord('a')  # Convert letter to 0-7
            row = int(move[1]) - 1  # Convert number to 0-7
            if 0 <= row < 8 and 0 <= col < 8:
                return row, col
            return None
        except (IndexError, ValueError):
            return None

    def numeric_to_algebraic(self, row, col):
        """
        Convert board coordinates to algebraic notation.
        
        Args:
            row (int): Board row (0-7)
            col (int): Board column (0-7)
            
        Returns:
            str: Move in algebraic notation (e.g., 'e3')
        """
        return f"{chr(col + ord('a'))}{row + 1}"

    def display_board(self):
        """Print current board state to output."""
        print()
        for i, row in enumerate(self.board):
            print(f"{i + 1} {' '.join(row)}")
        print("  a b c d e f g h")

    def _would_flip(self, row, col, dr, dc, player):
        """
        Check if placing a disc at (row, col) would flip discs in the given direction.
        
        Args:
            row (int): Target row
            col (int): Target column
            dr (int): Direction row delta
            dc (int): Direction column delta
            player (Player): Current player
            
        Returns:
            bool: True if discs would be flipped, False otherwise
        """
        r, c = row + dr, col + dc
        if not (0 <= r < 8 and 0 <= c < 8):
            return False
        if self.board[r][c] != player.opponent().value:
            return False
            
        while 0 <= r < 8 and 0 <= c < 8:
            if self.board[r][c] == ".":
                return False
            if self.board[r][c] == player.value:
                return True
            r, c = r + dr, c + dc
        return False

    def is_valid_move(self, row, col, player):
        """
        Check if a move is valid according to Othello rules.
        
        A valid move must:
        1. Be on an empty space
        2. Flank opponent's pieces
        3. Result in at least one piece being flipped
        
        Args:
            row (int): Target row
            col (int): Target column
            player (Player): Player making the move
            
        Returns:
            bool: True if move is valid, False otherwise
        """
        # Check if position is on board and empty
        if not (0 <= row < 8 and 0 <= col < 8) or self.board[row][col] != ".":
            return False
            
        # Check all 8 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc, player):
                return True
        return False

    def get_valid_moves(self, player):
        """
        Find all valid moves for the given player.
        
        Args:
            player (Player): Player to check moves for
            
        Returns:
            list: List of (row, col) tuples representing valid moves
        """
        valid_moves = []
        for r in range(8):
            for c in range(8):
                if self.is_valid_move(r, c, player):
                    valid_moves.append((r, c))
        return valid_moves

    def apply_move(self, row, col, player):
        """
        Apply a move to the board, flipping appropriate pieces.
        
        Args:
            row (int): Target row
            col (int): Target column
            player (Player): Player making the move
            
        Returns:
            bool: True if move was successfully applied
        """
        if not self.is_valid_move(row, col, player):
            return False

        self.board[row][col] = player.value
        flipped = []
        
        # Check all 8 directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        # Flip pieces in each valid direction
        for dr, dc in directions:
            r, c = row + dr, col + dc
            path = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player.opponent().value:
                path.append((r, c))
                r, c = r + dr, c + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player.value:
                flipped.extend(path)
                for pr, pc in path:
                    self.board[pr][pc] = player.value

        # Record move details for history
        flipped_positions = [self.numeric_to_algebraic(r, c) for r, c in flipped]
        move_details = {
            'player': player.value,
            'move': self.numeric_to_algebraic(row, col),
            'flipped': len(flipped),
            'flipped_positions': flipped_positions
        }
        self.move_history.append(move_details)

        # Update game log with detailed move information
        x_count, o_count = self.count_discs()
        move_log = {
            'move_number': len(self.move_history),
            'timestamp': datetime.now().isoformat(),
            'player': player.value,
            'move_position': self.numeric_to_algebraic(row, col),
            'flipped_discs': {
                'count': len(flipped),
                'positions': flipped_positions
            },
            'board_state': self.get_board_state(),
            'valid_moves': [self.numeric_to_algebraic(r, c) 
                          for r, c in self.get_valid_moves(player.opponent())],
            'score': {
                'X': x_count,
                'O': o_count
            }
        }
        self.game_log['moves'].append(move_log)
        return True

    def has_valid_move(self, player):
        """Check if player has any valid moves available."""
        return bool(self.get_valid_moves(player))

    def switch_player(self):
        """Switch current player to opponent."""
        self.current_player = self.current_player.opponent()

    def count_discs(self):
        """
        Count the number of discs for each player.
        
        Returns:
            tuple: (X_count, O_count)
        """
        x_count = sum(row.count(Player.BLACK.value) for row in self.board)
        o_count = sum(row.count(Player.WHITE.value) for row in self.board)
        return x_count, o_count

    def is_game_over(self):
        """Check if game is over (no valid moves for either player)."""
        return not (self.has_valid_move(Player.BLACK) or self.has_valid_move(Player.WHITE))

    def print_game_status(self):
        """Display current game status including scores and last move."""
        x_count, o_count = self.count_discs()
        print(f"\nCurrent score - Black (X): {x_count}, White (O): {o_count}")
        if self.move_history:
            last_move = self.move_history[-1]
            flipped_str = ", ".join(last_move['flipped_positions'])
            print(f"Last move: {last_move['player']} played {last_move['move']}, "
                  f"flipping {last_move['flipped']} disc(s) at: [{flipped_str}]")

    def save_game_log(self, filename=None):
        """
        Save complete game log to JSON file.
        
        Args:
            filename (str, optional): Custom filename for log
        """
        if self.game_log['moves']:  # Only save if moves were made
            self.game_log['end_time'] = datetime.now().isoformat()
            x_count, o_count = self.count_discs()
            self.game_log['final_score'] = {'X': x_count, 'O': o_count}
            self.game_log['winner'] = ('X' if x_count > o_count else 
                                     'O' if o_count > x_count else 'Tie')
            
            if filename is None:
                filename = f"othello_game_{self.game_log['game_id']}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.game_log, f, indent=2)
            print(f"\nGame log saved to {filename}")


class OthelloAI:
    """
    AI player implementation using minimax algorithm with alpha-beta pruning.
    
    Attributes:
        player (Player): AI's piece color (BLACK or WHITE)
        opponent (Player): Opponent's piece color
        depth (int): Search depth for minimax
        heuristic (int): Chosen evaluation function (1-3)
        nodes_evaluated (int): Number of nodes evaluated in search
        start_time (float): Start time of move calculation
    """
    def __init__(self, player, depth, heuristic):
        """Initialize AI player with specified settings."""
        self.player = player
        self.opponent = player.opponent()
        self.depth = depth
        self.heuristic = heuristic
        self.nodes_evaluated = 0
        self.start_time = 0

    def evaluate(self, game):
        """
        Evaluate board position using selected heuristic.
        Use h1 (disc count difference) for depths greater than 6.
        """
        if self.depth > 6:
            # Use h1 for high depths
            x_count, o_count = game.count_discs()
            return x_count - o_count if self.player == Player.BLACK else o_count - x_count
        else:
            # Use the selected heuristic for lower depths
            if self.heuristic == 1:
                x_count, o_count = game.count_discs()
                return x_count - o_count if self.player == Player.BLACK else o_count - x_count
            elif self.heuristic == 2:
                return self.positional_strategy(game)
            elif self.heuristic == 3:
                return self.mobility_stability(game)

    def positional_strategy(self, game):
        """
        Evaluate position based on weighted board positions.
        Corner positions are highly valued, edges are moderately valued,
        and positions next to corners are penalized.
        """
        # Position weights matrix
        weights = [
            [100, -20, 10,  5,  5, 10, -20, 100],
            [-20, -50,  1,  1,  1,  1, -50, -20],
            [ 10,   1,  5,  2,  2,  5,   1,  10],
            [  5,   1,  2,  1,  1,  2,   1,   5],
            [  5,   1,  2,  1,  1,  2,   1,   5],
            [ 10,   1,  5,  2,  2,  5,   1,  10],
            [-20, -50,  1,  1,  1,  1, -50, -20],
            [100, -20, 10,  5,  5, 10, -20, 100]
        ]
        
        score = 0
        for i in range(8):
            for j in range(8):
                if game.board[i][j] == self.player.value:
                    score += weights[i][j]
                elif game.board[i][j] == self.opponent.value:
                    score -= weights[i][j]
        return score

    def mobility_stability(self, game):
        """
        Evaluate position based on movement options and piece stability.
        Combines number of valid moves with stability of corner pieces.
        """
        # Calculate mobility (available moves)
        mobility_score = len(game.get_valid_moves(self.player)) - len(game.get_valid_moves(self.opponent))
        
        # Calculate stability (corner control and adjacent stable pieces)
        stability_score = self.count_stable_discs(game)
        
        # Combine scores with weights (mobility * 10 + stability * 30)
        return mobility_score * 10 + stability_score * 30

    def count_stable_discs(self, game):
        """
        Count number of stable discs (corners and adjacent stable pieces).
        Corners are most stable, followed by pieces adjacent to stable corners.
        
        Args:
            game (Othello): Current game state
            
        Returns:
            float: Stability score
        """
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        stable_count = 0
        
        for corner in corners:
            if game.board[corner[0]][corner[1]] == self.player.value:
                stable_count += 1
                # Check adjacent positions for additional stability
                for dr, dc in [(0, 1), (1, 0)] if corner[0] == 0 else [(0, -1), (-1, 0)]:
                    r, c = corner[0] + dr, corner[1] + dc
                    if 0 <= r < 8 and 0 <= c < 8 and game.board[r][c] == self.player.value:
                        stable_count += 0.5
                        
        return stable_count

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        """
        Implement minimax algorithm with alpha-beta pruning.
        
        Args:
            game (Othello): Current game state
            depth (int): Remaining search depth
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing, False if minimizing
            
        Returns:
            float: Best evaluation score for this position
        """
        self.nodes_evaluated += 1
        
        # Base cases: leaf node or game over
        if depth == 0 or game.is_game_over():
            return self.evaluate(game)

        valid_moves = game.get_valid_moves(self.player if maximizing_player else self.opponent)
        
        # If no valid moves, pass turn to opponent
        if not valid_moves:
            return self.minimax(game, depth - 1, alpha, beta, not maximizing_player)

        if maximizing_player:
            max_eval = float("-inf")
            for r, c in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.apply_move(r, c, self.player)
                eval = self.minimax(new_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float("inf")
            for r, c in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.apply_move(r, c, self.opponent)
                eval = self.minimax(new_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def find_best_move(self, game):
        """
        Find the best move for the AI using minimax search.
        
        Args:
            game (Othello): Current game state
            
        Returns:
            tuple: (row, col) of best move
        """
        self.nodes_evaluated = 0
        self.start_time = time.time()
        
        best_move = None
        best_value = float("-inf")
        valid_moves = game.get_valid_moves(self.player)
        
        # Evaluate each possible move
        for r, c in valid_moves:
            new_game = copy.deepcopy(game)
            new_game.apply_move(r, c, self.player)
            move_value = self.minimax(new_game, self.depth - 1, float("-inf"), float("inf"), False)
            if move_value > best_value:
                best_value = move_value
                best_move = (r, c)

        # Print AI performance statistics
        elapsed_time = time.time() - self.start_time
        print(f"\nAI Statistics:")
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        if elapsed_time > 0:
            print(f"Nodes per second: {self.nodes_evaluated / elapsed_time:.0f}")
        else:
            print("Nodes per second: N/A")
        
        return best_move


def start_ffmpeg_recording(output_filename):
    """Start FFmpeg to record the entire screen with cropping."""
    return subprocess.Popen([
        "ffmpeg", "-y", "-f", "gdigrab", "-framerate", "10", "-i", "desktop",
        "-filter_complex", "crop=1920:1032:0:0", "-c:v", "libx264", "-preset", "ultrafast", output_filename
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_ffmpeg_recording(ffmpeg_process):
    """Stops the FFmpeg recording process."""
    ffmpeg_process.stdin.write(b"q\n")
    ffmpeg_process.stdin.flush()
    ffmpeg_process.terminate()
    ffmpeg_process.wait()

def main():
    """
    Main game loop and program entry point.
    Handles game setup, mode selection, and gameplay flow.
    """
    # Save original stdout for restoration later
    original_stdout = sys.stdout
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # video_filename = f"othello_gameplay_{timestamp}.mp4"
    
    # Start FFmpeg recording
    # ffmpeg_process = start_ffmpeg_recording(video_filename)
    # print(f"Recording started...!")

    try:
        # Display game mode selection
        print("Welcome to Othello!")
        print("Choose Game Mode:")
        print("1. Human vs Human")
        print("2. Human vs AI")
        print("3. AI vs AI")

        # Get valid game mode selection
        while True:
            try:
                mode = int(input("Enter your choice (1-3): "))
                if mode in [1, 2, 3]:
                    break
                print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # Initialize variables for filename - AI settings if needed
        depth1 = 0
        depth2 = 0
        heuristic1 = 0
        heuristic2 = 0
        
        if mode == 2:
            # Get AI search depth for single AI player
            while True:
                try:
                    depth1 = int(input("Enter AI search depth (1-8 recommended): "))
                    if 1 <= depth1 <= 10:
                        break
                    print("Please enter a reasonable depth (1-8).")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Select AI heuristic
            print("\nAvailable heuristics:")
            print("1. Basic disc count")
            print("2. Positional strategy (weighted positions)")
            print("3. Combined mobility and stability")
            
            heuristic1 = int(input("Choose heuristic for AI (1-3): "))
            ai_player1 = OthelloAI(Player.WHITE, depth=depth1, heuristic=heuristic1)

        elif mode == 3:
            # Get AI search depth for first AI player
            while True:
                try:
                    depth1 = int(input("Enter search depth for AI 1 (Black/X) (1-8 recommended): "))
                    if 1 <= depth1 <= 10:
                        break
                    print("Please enter a reasonable depth (1-8).")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Get AI search depth for second AI player
            while True:
                try:
                    depth2 = int(input("Enter search depth for AI 2 (White/O) (1-8 recommended): "))
                    if 1 <= depth2 <= 10:
                        break
                    print("Please enter a reasonable depth (1-8).")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Select AI heuristics
            print("\nAvailable heuristics:")
            print("1. Basic disc count")
            print("2. Positional strategy (weighted positions)")
            print("3. Combined mobility and stability")
            
            heuristic1 = int(input("Choose heuristic for AI 1 (Black/X) (1-3): "))
            heuristic2 = int(input("Choose heuristic for AI 2 (White/O) (1-3): "))
            
            ai_player1 = OthelloAI(Player.BLACK, depth=depth1, heuristic=heuristic1)
            ai_player2 = OthelloAI(Player.WHITE, depth=depth2, heuristic=heuristic2)

        # Create descriptive filename based on game settings
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if mode == 1:
            output_filename = f"othello_HumanVsHuman_{timestamp}.txt"
        elif mode == 2:
            output_filename = f"othello_HumanVsAI_d{depth1}_h{heuristic1}_{timestamp}.txt"
        else:
            output_filename = f"othello_AIvsAI_d{depth1}d{depth2}_h{heuristic1}h{heuristic2}_{timestamp}.txt"

        # Set up output logging
        output_file = open(output_filename, 'w')
        sys.stdout = Tee(sys.stdout, output_file)

        # Initialize game and game log
        game = Othello()
        game_mode_str = {1: "Human vs Human", 2: "Human vs AI", 3: "AI vs AI"}[mode]
        game.game_log['game_mode'] = game_mode_str
        
        # Record AI settings in game log
        if mode == 2:
            game.game_log['ai_settings'] = {
                'AI1': {
                    'player': Player.WHITE.value,
                    'depth': depth1,
                    'heuristic': heuristic1
                }
            }
        elif mode == 3:
            game.game_log['ai_settings'] = {
                'AI1': {
                    'player': Player.BLACK.value,
                    'depth': depth1,
                    'heuristic': heuristic1
                },
                'AI2': {
                    'player': Player.WHITE.value,
                    'depth': depth2,
                    'heuristic': heuristic2
                }
            }

        # Main game loop
        while not game.is_game_over():
            game.display_board()
            game.print_game_status()
            
            # Handle player with no valid moves
            if not game.has_valid_move(game.current_player):
                print(f"\n{game.current_player.value} has no valid moves. Passing...")
                game.switch_player()
                continue

            # Handle moves based on game mode
            if mode == 1 or (mode == 2 and game.current_player == Player.BLACK):
                # Human player move
                print(f"\n{game.current_player.value}'s turn:")
                moves = game.get_valid_moves(game.current_player)
                print("Available moves:", 
                    [game.numeric_to_algebraic(r, c) for r, c in moves])
                
                # Get valid move from human player
                while True:
                    move = input("Enter your move (e.g., 'e3'): ").strip().lower()
                    coords = game.algebraic_to_numeric(move)
                    if coords and coords in moves:
                        game.apply_move(coords[0], coords[1], game.current_player)
                        break
                    print("Invalid move. Please try again.")
                    
            elif mode == 2 and game.current_player == Player.WHITE:
                # AI player move in Human vs AI mode
                print("\nAI's turn (O):")
                move = ai_player1.find_best_move(game)
                if move:
                    game.apply_move(move[0], move[1], Player.WHITE)
                    print(f"AI plays: {game.numeric_to_algebraic(move[0], move[1])}")
                    
            elif mode == 3:
                # AI vs AI mode
                print(f"\nAI {game.current_player.value}'s turn:")
                ai_player = ai_player1 if game.current_player == Player.BLACK else ai_player2
                move = ai_player.find_best_move(game)
                if move:
                    game.apply_move(move[0], move[1], game.current_player)
                    print(f"AI plays: {game.numeric_to_algebraic(move[0], move[1])}")
            
            game.switch_player()

        # Game over - display final results
        game.display_board()
        x_count, o_count = game.count_discs()
        print("\nGame Over!")
        print(f"Final score - Black (X): {x_count}, White (O): {o_count}")
        
        if x_count > o_count:
            print("Black (X) wins!")
        elif o_count > x_count:
            print("White (O) wins!")
        else:
            print("It's a tie!")

        # Save game log and clean up
        game.save_game_log()

    finally:
        # Add delay to ensure FFmpeg process is stopped -- To View last frame of the game in a clean way
        # time.sleep(5)
        # Stop video recording and restore stdout
        # stop_ffmpeg_recording(ffmpeg_process)
        # Restore original stdout and close file
        sys.stdout = original_stdout
        output_file.close()
        print(f"\nGame output has been saved to {output_filename}")
        # print(f"\nGame recording have been saved as {video_filename}")


if __name__ == "__main__":
    main()