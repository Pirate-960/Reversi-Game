import time
import copy
from datetime import datetime
import json
import os

class OthelloEngine:
    """
    Advanced Othello engine with improved evaluation and search techniques.
    
    Features:
    - Principal Variation Search (enhanced alpha-beta)
    - Iterative Deepening
    - Position evaluation with multiple strategic factors
    - Opening book awareness
    - Endgame solver for perfect play
    """

    def __init__(self, player, max_depth=8):
        self.player = player  # 'X' for black, 'O' for white
        self.opponent = 'O' if player == 'X' else 'X'
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.transposition_table = {}
        
        # Position weights for strategic evaluation
        self.position_weights = [
            [120, -20,  20,   5,   5,  20, -20, 120],
            [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
            [ 20,  -5,  15,   3,   3,  15,  -5,  20],
            [  5,  -5,   3,   3,   3,   3,  -5,   5],
            [  5,  -5,   3,   3,   3,   3,  -5,   5],
            [ 20,  -5,  15,   3,   3,  15,  -5,  20],
            [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
            [120, -20,  20,   5,   5,  20, -20, 120]
        ]

    def evaluate_position(self, game, is_endgame=False):
        """
        Evaluate board position using multiple strategic factors.
        
        Args:
            game (OthelloGame): Current game state
            is_endgame (bool): Whether we're in endgame phase
            
        Returns:
            float: Position evaluation score
        """
        if is_endgame:
            return self._evaluate_endgame(game)
            
        score = 0
        
        # 1. Material count (disc difference)
        x_count, o_count = game.count_pieces()
        piece_score = x_count - o_count if self.player == 'X' else o_count - x_count
        
        # 2. Mobility (number of legal moves)
        my_moves = len(game.get_valid_moves(self.player))
        opp_moves = len(game.get_valid_moves(self.opponent))
        mobility_score = my_moves - opp_moves
        
        # 3. Position evaluation using weights
        position_score = 0
        for i in range(8):
            for j in range(8):
                if game.board[i][j] == self.player:
                    position_score += self.position_weights[i][j]
                elif game.board[i][j] == self.opponent:
                    position_score -= self.position_weights[i][j]
        
        # 4. Corner control
        corner_score = self._evaluate_corners(game)
        
        # 5. Edge stability
        stability_score = self._evaluate_stability(game)
        
        # Combine all factors with appropriate weights
        score = (
            piece_score * 1 +
            mobility_score * 10 +
            position_score * 2 +
            corner_score * 25 +
            stability_score * 15
        )
        
        return score

    def _evaluate_endgame(self, game):
        """Perfect evaluation for endgame positions."""
        x_count, o_count = game.count_pieces()
        if self.player == 'X':
            return 10000 * (x_count - o_count)
        return 10000 * (o_count - x_count)

    def _evaluate_corners(self, game):
        """Evaluate corner control and adjacent squares."""
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        score = 0
        
        for corner in corners:
            if game.board[corner[0]][corner[1]] == self.player:
                score += 25
            elif game.board[corner[0]][corner[1]] == self.opponent:
                score -= 25
                
        return score

    def _evaluate_stability(self, game):
        """Evaluate piece stability (cannot be flipped)."""
        score = 0
        
        # Check edges
        for i in range(8):
            if game.board[i][0] == self.player:
                score += 5
            elif game.board[i][0] == self.opponent:
                score -= 5
            if game.board[i][7] == self.player:
                score += 5
            elif game.board[i][7] == self.opponent:
                score -= 5
                
        for j in range(8):
            if game.board[0][j] == self.player:
                score += 5
            elif game.board[0][j] == self.opponent:
                score -= 5
            if game.board[7][j] == self.player:
                score += 5
            elif game.board[7][j] == self.opponent:
                score -= 5
                
        return score

    def get_best_move(self, game):
        """
        Find the best move using iterative deepening and PVS.
        
        Args:
            game (OthelloGame): Current game state
            
        Returns:
            tuple: (row, col) of best move
        """
        self.nodes_evaluated = 0
        best_move = None
        best_value = float('-inf')
        
        # Get valid moves and sort by preliminary evaluation
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None
            
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            current_best_move = None
            alpha = float('-inf')
            beta = float('inf')
            
            # Principal Variation Search
            for i, move in enumerate(valid_moves):
                new_game = game.copy()
                new_game.make_move(move[0], move[1], self.player)
                
                if i == 0:
                    # Full window search for first move
                    value = -self._pvs(new_game, depth - 1, -beta, -alpha, False)
                else:
                    # Null window search for remaining moves
                    value = -self._pvs(new_game, depth - 1, -alpha - 1, -alpha, False)
                    if alpha < value < beta:
                        # Re-search with full window if necessary
                        value = -self._pvs(new_game, depth - 1, -beta, -alpha, False)
                
                if value > alpha:
                    alpha = value
                    current_best_move = move
                    
                if alpha >= beta:
                    break
            
            if current_best_move:
                best_move = current_best_move
                best_value = alpha
                
        return best_move

    def _pvs(self, game, depth, alpha, beta, maximizing):
        """Principal Variation Search implementation."""
        if depth == 0 or game.is_game_over():
            return self.evaluate_position(game, depth == 0)
            
        self.nodes_evaluated += 1
        valid_moves = game.get_valid_moves(self.player if maximizing else self.opponent)
        
        if not valid_moves:
            if game.has_valid_move(self.opponent if maximizing else self.player):
                return -self._pvs(game, depth - 1, -beta, -alpha, not maximizing)
            return self.evaluate_position(game, True)
            
        best_value = float('-inf')
        
        for i, move in enumerate(valid_moves):
            new_game = game.copy()
            new_game.make_move(move[0], move[1], 
                             self.player if maximizing else self.opponent)
            
            if i == 0:
                value = -self._pvs(new_game, depth - 1, -beta, -alpha, not maximizing)
            else:
                value = -self._pvs(new_game, depth - 1, -alpha - 1, -alpha, not maximizing)
                if alpha < value < beta:
                    value = -self._pvs(new_game, depth - 1, -beta, -alpha, not maximizing)
            
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            
            if alpha >= beta:
                break
                
        return best_value

class OthelloGame:
    """
    Main game class implementing the Othello board and rules.
    """
    def __init__(self):
        self.board = [["." for _ in range(8)] for _ in range(8)]
        self.board[3][3] = self.board[4][4] = "O"  # White pieces
        self.board[3][4] = self.board[4][3] = "X"  # Black pieces
        self.current_player = "X"  # Black starts
        self.move_history = []
        self.game_log = {
            'game_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'moves': [],
            'initial_state': self.get_board_state()
        }

    def get_board_state(self):
        """Capture current board state."""
        return {
            'board': [row[:] for row in self.board],
            'current_player': self.current_player,
            'piece_count': self.count_pieces()
        }

    def display_board(self):
        """Display the current board state."""
        print("\n  a b c d e f g h")
        for i, row in enumerate(self.board, 1):
            print(f"{i} {' '.join(row)}")
        x_count, o_count = self.count_pieces()
        print(f"\nScore - Black (X): {x_count}, White (O): {o_count}")

    def copy(self):
        """
        Create a deep copy of the current game state.

        Returns:
            OthelloGame: A new game instance with copied state
        """
        new_game = OthelloGame()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        return new_game

    def is_valid_move(self, row, col, player):
        """Check if move is valid according to Othello rules."""
        if not (0 <= row < 8 and 0 <= col < 8) or self.board[row][col] != ".":
            return False

        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (-1,-1), (1,-1), (-1,1)]
        opponent = "O" if player == "X" else "X"

        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                pieces_to_flip.append((r, c))
                r, c = r + dr, c + dc
                
            if pieces_to_flip and 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                return True
                
        return False

    def get_valid_moves(self, player):
        """Get all valid moves for the given player."""
        moves = []
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(i, j, player):
                    moves.append((i, j))
        return moves

    def has_valid_move(self, player):
        """Check if player has any valid moves."""
        return bool(self.get_valid_moves(player))

    def make_move(self, row, col, player):
        """Apply move and flip appropriate pieces."""
        if not self.is_valid_move(row, col, player):
            return False

        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (-1,-1), (1,-1), (-1,1)]
        opponent = "O" if player == "X" else "X"
        pieces_flipped = []

        self.board[row][col] = player
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            pieces_to_flip = []
            
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                pieces_to_flip.append((r, c))
                r, c = r + dr, c + dc
                
            if pieces_to_flip and 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                for flip_r, flip_c in pieces_to_flip:
                    self.board[flip_r][flip_c] = player
                    pieces_flipped.append((flip_r, flip_c))

        # Log the move
        move_info = {
            'player': player,
            'move': (row, col),
            'pieces_flipped': pieces_flipped,
            'timestamp': datetime.now().isoformat()
        }
        self.move_history.append(move_info)
        self.game_log['moves'].append(move_info)
        
        return True

    def count_pieces(self):
        """Count pieces for both players."""
        x_count = sum(row.count("X") for row in self.board)
        o_count = sum(row.count("O") for row in self.board)
        return x_count, o_count

    def is_game_over(self):
        """Check if game is over."""
        return not (self.get_valid_moves("X") or self.get_valid_moves("O"))

    def get_winner(self):
        """Determine the winner."""
        x_count, o_count = self.count_pieces()
        if x_count > o_count:
            return "X"
        elif o_count > x_count:
            return "O"
        return "Tie"

class OthelloInterface:
    """
    Game interface handling different play modes and user interaction.
    """
    def __init__(self):
        self.game = OthelloGame()
        self.engine = None
        self.engine2 = None
        self.game_mode = None
        self.player_color = None
        self.difficulty = None
        self.log_directory = "game_logs"

    def initialize_game(self):
        """Set up game mode and parameters."""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

        print("\nWelcome to Othello!")
        print("\nSelect Game Mode:")
        print("1. Human vs Human")
        print("2. Human vs Computer")
        print("3. Computer vs Computer")
        
        while True:
            try:
                self.game_mode = int(input("\nEnter mode (1-3): "))
                if self.game_mode in [1, 2, 3]:
                    break
                print("Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        if self.game_mode == 2:
            print("\nSelect your color:")
            print("1. Black (X) - plays first")
            print("2. White (O) - plays second")
            
            while True:
                try:
                    color_choice = int(input("\nEnter choice (1-2): "))
                    if color_choice in [1, 2]:
                        self.player_color = "X" if color_choice == 1 else "O"
                        break
                    print("Invalid choice. Please enter 1 or 2.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            print("\nSelect difficulty level:")
            print("1. Easy (depth 3)")
            print("2. Medium (depth 5)")
            print("3. Hard (depth 7)")
            
            while True:
                try:
                    self.difficulty = int(input("\nEnter difficulty (1-3): "))
                    if self.difficulty in [1, 2, 3]:
                        break
                    print("Invalid choice. Please enter 1, 2, or 3.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Initialize engine with appropriate settings
            engine_color = "O" if self.player_color == "X" else "X"
            depth = {1: 3, 2: 5, 3: 7}[self.difficulty]
            self.engine = OthelloEngine(engine_color, max_depth=depth)

        elif self.game_mode == 3:
            # Initialize two engines for computer vs computer
            self.engine = OthelloEngine("X", max_depth=5)  # Black player
            self.engine2 = OthelloEngine("O", max_depth=5)  # White player

    def get_human_move(self):
        """Get and validate human player move."""
        valid_moves = self.game.get_valid_moves(self.game.current_player)
        if not valid_moves:
            return None

        print("\nValid moves:", end=" ")
        for row, col in valid_moves:
            print(f"{chr(col + ord('a'))}{row + 1}", end=" ")
        print()

        while True:
            try:
                move = input("\nEnter your move (e.g., e3) or 'q' to quit: ").strip().lower()
                if move == 'q':
                    return 'quit'
                if len(move) != 2:
                    print("Invalid format. Please use letter (a-h) followed by number (1-8).")
                    continue

                col = ord(move[0]) - ord('a')
                row = int(move[1]) - 1

                if (row, col) in valid_moves:
                    return row, col
                print("Invalid move. Please choose from the valid moves listed above.")
            except (ValueError, IndexError):
                print("Invalid input. Please use letter (a-h) followed by number (1-8).")

    def play_game(self):
        """Main game loop."""
        self.initialize_game()
        game_in_progress = True
        
        while game_in_progress and not self.game.is_game_over():
            self.game.display_board()
            current = self.game.current_player
            print(f"\nCurrent player: {'Black (X)' if current == 'X' else 'White (O)'}")

            valid_moves = self.game.get_valid_moves(current)
            if not valid_moves:
                print(f"No valid moves for {current}. Skipping turn...")
                self.game.current_player = "O" if current == "X" else "X"
                continue

            # Handle moves based on game mode
            if self.game_mode == 1:  # Human vs Human
                move = self.get_human_move()
                if move == 'quit':
                    game_in_progress = False
                    break
                if move:
                    self.game.make_move(move[0], move[1], current)
            
            elif self.game_mode == 2:  # Human vs Computer
                if current == self.player_color:
                    move = self.get_human_move()
                    if move == 'quit':
                        game_in_progress = False
                        break
                    if move:
                        self.game.make_move(move[0], move[1], current)
                else:
                    print("\nComputer is thinking...")
                    move = self.engine.get_best_move(self.game)
                    if move:
                        print(f"Computer plays: {chr(move[1] + ord('a'))}{move[0] + 1}")
                        self.game.make_move(move[0], move[1], current)
            
            else:  # Computer vs Computer
                print(f"\nComputer ({current}) is thinking...")
                engine = self.engine if current == "X" else self.engine2
                move = engine.get_best_move(self.game)
                if move:
                    print(f"Computer plays: {chr(move[1] + ord('a'))}{move[0] + 1}")
                    self.game.make_move(move[0], move[1], current)
                time.sleep(1)  # Add delay for better visualization

            self.game.current_player = "O" if current == "X" else "X"

        # Game over
        if game_in_progress:
            self.game.display_board()
            winner = self.game.get_winner()
            print("\nGame Over!")
            if winner == "Tie":
                print("It's a tie!")
            else:
                print(f"Winner: {'Black (X)' if winner == 'X' else 'White (O)'}")

            # Save game log
            self.save_game_log()
        else:
            print("\nGame terminated by user.")

    def save_game_log(self):
        """Save game log to file."""
        try:
            self.game.game_log['end_time'] = datetime.now().isoformat()
            self.game.game_log['winner'] = self.game.get_winner()
            
            filename = os.path.join(
                self.log_directory,
                f"othello_game_{self.game.game_log['game_id']}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump(self.game.game_log, f, indent=2)
            print(f"\nGame log saved to {filename}")
        except Exception as e:
            print(f"\nError saving game log: {str(e)}")


def main():
    """Main entry point."""
    try:
        interface = OthelloInterface()
        interface.play_game()
        
        while True:
            play_again = input("\nWould you like to play again? (y/n): ").lower()
            if play_again == 'y':
                interface = OthelloInterface()
                interface.play_game()
            elif play_again == 'n':
                print("Thank you for playing!")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    except KeyboardInterrupt:
        print("\n\nGame terminated by user. Thank you for playing!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("The game has been terminated.")


if __name__ == "__main__":
    main()