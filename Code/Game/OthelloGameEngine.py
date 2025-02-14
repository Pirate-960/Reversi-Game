import copy
import time
import json
import sys
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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

class Patterns:
    """Pre-computed pattern tables for evaluation."""
    # Corner patterns and their values
    CORNER_PATTERNS = {
        # Empty corner patterns
        ".XXX": 50,  "XXX.": 50,  # Three in a row with empty corner
        ".XX": 30,   "XX.": 30,   # Two in a row with empty corner
        ".X": 10,    "X.": 10,    # One with empty corner
        
        # Occupied corner patterns
        "XXXX": 100, # Complete line control
        "XXX": 80,   # Three with corner
        "XX": 60,    # Two with corner
        "X": 40      # Corner occupied
    }
    
    # Edge patterns and values
    EDGE_PATTERNS = {
        "XXXX": 40,  # Complete edge control
        ".XXX": 20,  # Three in a row
        "XX..": 10,  # Two with space
        "X..X": 15   # Separated pieces
    }

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_pattern_value(pattern: str, player_value: str) -> int:
        """Get value for a pattern, accounting for player color."""
        # Normalize pattern to X-based representation
        if player_value == "O":
            pattern = pattern.replace("X", "T").replace("O", "X").replace("T", "O")
            
        # Check corner patterns
        if pattern in Patterns.CORNER_PATTERNS:
            return Patterns.CORNER_PATTERNS[pattern]
            
        # Check edge patterns
        if pattern in Patterns.EDGE_PATTERNS:
            return Patterns.EDGE_PATTERNS[pattern]
            
        return 0

class AdvancedEvaluation:
    """Advanced position evaluation with pattern matching and strategic concepts."""
    
    def __init__(self):
        self.pattern_cache = {}
        self.mobility_cache = {}
        self.stability_cache = {}
        
    @staticmethod
    def get_disc_count(board: List[List[str]], player_value: str) -> Tuple[int, int]:
        """Get disc count difference between player and opponent."""
        player_count = sum(row.count(player_value) for row in board)
        opponent_value = "O" if player_value == "X" else "X"
        opponent_count = sum(row.count(opponent_value) for row in board)
        return player_count, opponent_count

    @staticmethod
    @lru_cache(maxsize=1024)
    def get_potential_mobility(board: List[List[str]], row: int, col: int) -> int:
        """Calculate potential mobility score for a position."""
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        score = 0
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == ".":
                score += 1
                
        return score

    def evaluate_patterns(self, board: List[List[str]], player_value: str) -> int:
        """Evaluate board patterns focusing on corners and edges."""
        score = 0
        opponent_value = "O" if player_value == "X" else "X"
        
        # Corner evaluation
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner_row, corner_col in corners:
            # Horizontal pattern from corner
            if corner_col == 0:
                pattern = "".join(board[corner_row][0:4])
                score += Patterns.get_pattern_value(pattern, player_value)
            else:
                pattern = "".join(board[corner_row][4:8])
                score += Patterns.get_pattern_value(pattern[::-1], player_value)
                
            # Vertical pattern from corner
            if corner_row == 0:
                pattern = "".join(board[i][corner_col] for i in range(4))
                score += Patterns.get_pattern_value(pattern, player_value)
            else:
                pattern = "".join(board[i][corner_col] for i in range(4,8))
                score += Patterns.get_pattern_value(pattern[::-1], player_value)
                
        return score

    def evaluate_mobility(self, board: List[List[str]], valid_moves: List[Tuple[int, int]], 
                         opponent_moves: List[Tuple[int, int]]) -> int:
        """Evaluate current and potential mobility."""
        current_mobility = len(valid_moves) - len(opponent_moves)
        potential_mobility = 0
        
        # Calculate potential mobility for empty squares
        for row in range(8):
            for col in range(8):
                if board[row][col] == ".":
                    potential_mobility += self.get_potential_mobility(tuple(map(tuple, board)), row, col)
                    
        return current_mobility * 10 + potential_mobility * 5

    def evaluate_stability(self, board: List[List[str]], player_value: str) -> int:
        """Evaluate piece stability using sophisticated analysis."""
        opponent_value = "O" if player_value == "X" else "X"
        score = 0
        stable_pieces = set()
        
        # Start with corners
        corners = [(0,0), (0,7), (7,0), (7,7)]
        for corner in corners:
            if board[corner[0]][corner[1]] == player_value:
                stable_pieces.add(corner)
                
        # Expand stability from stable pieces
        old_size = 0
        while len(stable_pieces) > old_size:
            old_size = len(stable_pieces)
            for row in range(8):
                for col in range(8):
                    if (row, col) not in stable_pieces and board[row][col] == player_value:
                        if self._is_stable(board, row, col, stable_pieces):
                            stable_pieces.add((row, col))
                            
        return len(stable_pieces) * 30

    def _is_stable(self, board: List[List[str]], row: int, col: int, 
                  stable_pieces: set) -> bool:
        """Check if a piece is stable given current stable pieces."""
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        
        # Check if piece is protected in all directions
        for dr, dc in directions:
            protected = False
            r, c = row + dr, col + dc
            
            # Check if protected by board edge
            if not (0 <= r < 8 and 0 <= c < 8):
                protected = True
                continue
                
            # Check if protected by stable piece
            if (r, c) in stable_pieces:
                protected = True
                continue
                
            if not protected:
                return False
                
        return True

    def evaluate_parity(self, board: List[List[str]]) -> int:
        """Evaluate disc parity and empty square parity."""
        empty_count = sum(row.count(".") for row in board)
        return 10 if empty_count % 2 == 0 else -10

    def evaluate_position(self, board: List[List[str]], player: Player,
                         valid_moves: List[Tuple[int, int]], 
                         opponent_moves: List[Tuple[int, int]], 
                         game_phase: float) -> float:
        """
        Comprehensive position evaluation using multiple strategic components.
        
        Args:
            board: Current game board
            player: Current player
            valid_moves: List of valid moves for current player
            opponent_moves: List of valid moves for opponent
            game_phase: Game progression (0.0 to 1.0)
            
        Returns:
            float: Position evaluation score
        """
        player_value = player.value
        player_count, opponent_count = self.get_disc_count(board, player_value)
        total_discs = player_count + opponent_count
        
        # Early game: Focus on mobility and development
        if game_phase < 0.3:
            pattern_weight = 30
            mobility_weight = 50
            stability_weight = 10
            parity_weight = 10
            disc_weight = 0
            
        # Mid game: Balance between all factors
        elif game_phase < 0.7:
            pattern_weight = 40
            mobility_weight = 30
            stability_weight = 20
            parity_weight = 5
            disc_weight = 5
            
        # Late game: Focus on stability and disc count
        else:
            pattern_weight = 20
            mobility_weight = 10
            stability_weight = 40
            parity_weight = 10
            disc_weight = 20
            
        # Calculate component scores
        pattern_score = self.evaluate_patterns(board, player_value)
        mobility_score = self.evaluate_mobility(board, valid_moves, opponent_moves)
        stability_score = self.evaluate_stability(board, player_value)
        parity_score = self.evaluate_parity(board)
        disc_score = (player_count - opponent_count) * (100 if game_phase > 0.9 else 1)
        
        # Combine weighted components
        total_score = (pattern_score * pattern_weight +
                      mobility_score * mobility_weight +
                      stability_score * stability_weight +
                      parity_score * parity_weight +
                      disc_score * disc_weight) / 100.0
                      
        return total_score

class AdvancedOthelloAI:
    """Advanced AI player using enhanced search techniques."""
    
    def __init__(self, player: Player, depth: int):
        self.player = player
        self.opponent = player.opponent()
        self.max_depth = depth
        self.evaluator = AdvancedEvaluation()
        self.nodes_evaluated = 0
        self.tt_table = {}  # Transposition table
        self.history_table = {}  # Move ordering history
        self.killer_moves = [[None, None] for _ in range(64)]  # Two killer moves per ply
        
    def get_board_hash(self, board: List[List[str]]) -> int:
        """Generate a Zobrist hash of the board position."""
        return hash(tuple(tuple(row) for row in board))

    def order_moves(self, moves: List[Tuple[int, int]], depth: int, board_hash: int) -> List[Tuple[int, int]]:
        """Order moves using history heuristic and killer moves."""
        ordered_moves = []
        
        for move in moves:
            score = 0
            # History heuristic
            if (board_hash, move) in self.history_table:
                score += self.history_table[(board_hash, move)]
            
            # Killer move bonus
            if move in self.killer_moves[depth]:
                score += 10000
                
            ordered_moves.append((move, score))
            
        return [move for move, _ in sorted(ordered_moves, key=lambda x: x[1], reverse=True)]

    def update_history(self, board_hash: int, move: Tuple[int, int], depth: int):
        """Update history table with successful move."""
        key = (board_hash, move)
        self.history_table[key] = self.history_table.get(key, 0) + 2 ** depth

    def update_killer_moves(self, move: Tuple[int, int], depth: int):
        """Update killer moves for the current ply."""
        if move != self.killer_moves[depth][0]:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def negamax(self, game, depth: int, alpha: float, beta: float, color: int, 
                max_player: bool, null_move_allowed: bool = True) -> float:
        """
        Enhanced Negamax search with various improvements:
        - Null move pruning
        - Transposition table
        - Move ordering
        - Aspiration windows
        - Killer move heuristic
        - History heuristic
        """
        self.nodes_evaluated += 1
        board_hash = self.get_board_hash(game.board)
        alpha_orig = alpha
        
        # Transposition table lookup
        tt_entry = self.tt_table.get(board_hash)
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['value']
            elif tt_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_entry['value'])
            if alpha >= beta:
                return tt_entry['value']
        
        # Base cases
        if depth == 0 or game.is_game_over():
            player = self.player if max_player else self.opponent
            valid_moves = game.get_valid_moves(player)
            opponent_moves = game.get_valid_moves(player.opponent())
            game_phase = (sum(row.count("X") + row.count("O") for row in game.board) - 4) / 60.0
            
            score = self.evaluator.evaluate_position(
                game.board, player, valid_moves, opponent_moves, game_phase
            )
            return color * score
        
        # Null move pruning
        if null_move_allowed and depth >= 3 and not max_player:
            new_game = copy.deepcopy(game)
            new_game.switch_player()
            null_value = -self.negamax(new_game, depth-3, -beta, -beta+1, -color, 
                                     not max_player, False)
            if null_value >= beta:
                return beta
        
        valid_moves = game.get_valid_moves(self.player if max_player else self.opponent)
        
        # Handle no valid moves
        if not valid_moves:
            if not game.has_valid_move(self.player.opponent() if max_player else self.player):
                return color * float('inf') if max_player else color * float('-inf')
            new_game = copy.deepcopy(game)
            new_game.switch_player()
            return -self.negamax(new_game, depth-1, -beta, -alpha, -color, not max_player)
        
        # Move ordering
        ordered_moves = self.order_moves(valid_moves, depth, board_hash)
        best_value = float('-inf')
        best_move = None
        
        # Principal Variation Search
        for i, move in enumerate(ordered_moves):
            new_game = copy.deepcopy(game)
            new_game = copy.deepcopy(game)
            player = self.player if max_player else self.opponent
            new_game.apply_move(move[0], move[1], player)
            
            # Principal Variation Search
            if i == 0:
                value = -self.negamax(new_game, depth-1, -beta, -alpha, -color, not max_player)
            else:
                # Try null window search first
                value = -self.negamax(new_game, depth-1, -alpha-1, -alpha, -color, not max_player)
                if alpha < value < beta:
                    # Full-window search if null window fails
                    value = -self.negamax(new_game, depth-1, -beta, -alpha, -color, not max_player)
            
            if value > best_value:
                best_value = value
                best_move = move
                
            alpha = max(alpha, value)
            if alpha >= beta:
                # Update killer moves and history table for good moves
                self.update_killer_moves(move, depth)
                self.update_history(board_hash, move, depth)
                break
        
        # Store position in transposition table
        tt_entry = {
            'value': best_value,
            'depth': depth,
            'best_move': best_move,
            'flag': 'EXACT' if alpha_orig < best_value < beta else
                    'LOWERBOUND' if best_value >= beta else
                    'UPPERBOUND'
        }
        self.tt_table[board_hash] = tt_entry
        
        return best_value

    def find_best_move(self, game) -> Tuple[int, int]:
        """
        Find the best move using iterative deepening and aspiration windows.
    
        Returns:
            tuple: (row, col) of best move
        """
        self.nodes_evaluated = 0
        start_time = time.time()
    
        # Initialize search variables
        best_move = None
        best_value = float('-inf')
        current_depth = 1
    
        # Initialize aspiration window parameters
        window_size = 50
        alpha = float('-inf')
        beta = float('inf')
    
        # Clear transposition table for new search
        self.tt_table.clear()
    
        try:
            # Iterative deepening with aspiration windows
            while current_depth <= self.max_depth:
                self.nodes_evaluated = 0
                search_start = time.time()
            
                if current_depth > 1:
                    # Set aspiration windows based on previous search
                    alpha = max(best_value - window_size, float('-inf'))
                    beta = min(best_value + window_size, float('inf'))
            
                value = float('-inf')
                moves = game.get_valid_moves(self.player)
            
                if not moves:
                    break
                
                # Order moves for initial search
                ordered_moves = self.order_moves(moves, current_depth, self.get_board_hash(game.board))
            
                # Try aspiration window search
                try:
                    for move in ordered_moves:
                        new_game = copy.deepcopy(game)
                        new_game.apply_move(move[0], move[1], self.player)
                    
                        move_value = -self.negamax(new_game, current_depth-1, -beta, -alpha, -1, False)
                    
                        if move_value > value:
                            value = move_value
                            best_move = move
                        
                        alpha = max(alpha, value)
                        if alpha >= beta:
                            break
                        
                except ValueError:
                    # Window failed, retry with full window
                    alpha = float('-inf')
                    beta = float('inf')
                    continue
            
                best_value = value
                elapsed = time.time() - search_start
            
                # Print search statistics with safe division
                nps = self.nodes_evaluated / max(elapsed, 0.0001)  # Prevent division by zero
                print(f"Depth {current_depth:<3}: Value ={best_value:+12.2f} | Nodes ={self.nodes_evaluated:8} | Time ={elapsed:12.4f}s | NPS ={nps:8.0f}")
            
                # Check if we have enough time for next iteration
                if elapsed > 5000.0 and current_depth > 4:
                    # Stop search if we exceed 5000 seconds at depth > 4 - [more time -> more depth explored]
                    break
                
                # Increment depth for next iteration
                current_depth += 1
                # Adjust window size based on search results for next iteration (optional)
                window_size = max(50, abs(best_value) // 2)  # Adjust window size
            
        except KeyboardInterrupt:
            pass  # Allow manual interruption while keeping best move found
    
        return best_move


class Othello:
    """
    Enhanced Othello game class with advanced features and optimizations.
    Core game logic remains similar but with performance improvements.
    """
    def __init__(self):
        # Initialize board and game state (same as original)
        self.board = [["." for _ in range(8)] for _ in range(8)]
        self.board[3][3] = self.board[4][4] = Player.WHITE.value
        self.board[3][4] = self.board[4][3] = Player.BLACK.value
        self.current_player = Player.BLACK
        self.move_history = []
        self.game_log = {
            'game_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'moves': [],
            'initial_state': self.get_board_state()
        }
        
        # Pre-compute valid directions for each position
        self.valid_directions = {}
        for r in range(8):
            for c in range(8):
                self.valid_directions[(r, c)] = self._get_valid_directions(r, c)

    def _get_valid_directions(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Pre-compute valid directions for a position."""
        directions = []
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                directions.append((dr, dc))
        return directions

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
        Modified to use pre-computed valid directions.
        
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
        
        # Use pre-computed valid directions
        for dr, dc in self.valid_directions.get((row, col), []):
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
        
        # Check all potential directions
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]

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


def main():
    """
    Enhanced main game loop with advanced AI capabilities.
    """
    # Save original stdout for restoration later
    original_stdout = sys.stdout
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("Welcome to Advanced Othello!")
    print("Choose Game Mode:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    
    mode = int(input("Enter mode (1-3): "))
    
    if mode == 2:
        depth = int(input("Enter AI search depth (recommended 6-8): "))
        ai = AdvancedOthelloAI(Player.WHITE, depth)
    elif mode == 3:
        depth1 = int(input("Enter first AI depth (recommended 6-8): "))
        depth2 = int(input("Enter second AI depth (recommended 6-8): "))
        ai1 = AdvancedOthelloAI(Player.BLACK, depth1)
        ai2 = AdvancedOthelloAI(Player.WHITE, depth2)
    
    # Create descriptive filename based on game settings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if mode == 1:
        output_filename = f"othello_HumanVsHuman_{timestamp}.txt"
    elif mode == 2:
        output_filename = f"othello_HumanVsAI_d{depth}_{timestamp}.txt"
    else:
        output_filename = f"othello_AIvsAI_d{depth1}d{depth2}_{timestamp}.txt"

    # Set up output logging
    # Open file and set up Tee to write to both console and file
    output_file = open(output_filename, 'w')
    sys.stdout = Tee(sys.stdout, output_file)

    # Start the game
    game = Othello()
    
    # Main game loop until game is over or no valid moves available for both players
    while not game.is_game_over():
        game.display_board()
        game.print_game_status()
        
        # Check if current player has any valid moves available before proceeding further
        if not game.has_valid_move(game.current_player):
            print(f"\n{game.current_player.value} has no valid moves. Passing...")
            game.switch_player()
            continue
        
        # Get move from human or AI based on game mode and current player color
        if mode == 1 or (mode == 2 and game.current_player == Player.BLACK):
            # Human move
            moves = game.get_valid_moves(game.current_player)
            print("Valid moves:", [game.numeric_to_algebraic(r, c) for r, c in moves])
            while True:
                move = input("Enter move (e.g., 'e3'): ").strip().lower()
                coords = game.algebraic_to_numeric(move)
                if coords and coords in moves:
                    game.apply_move(coords[0], coords[1], game.current_player)
                    break
                print("Invalid move. Try again.")
        else:
            # AI move (Human vs AI or AI vs AI) - use appropriate AI instance based on player color
            print(f"\nAI {game.current_player.value} thinking...")
            if mode == 2:
                move = ai.find_best_move(game)
            else:
                ai = ai1 if game.current_player == Player.BLACK else ai2
                move = ai.find_best_move(game)
                
            if move:
                game.apply_move(move[0], move[1], game.current_player)
                print(f"AI plays: {game.numeric_to_algebraic(move[0], move[1])}")
        
        game.switch_player()
    
    # Game over - display final board and scores
    game.display_board()
    x_count, o_count = game.count_discs()
    print("\nGame Over!")
    print(f"Final score - Black: {x_count}, White: {o_count}")
    game.save_game_log()

    # Restore original stdout and close file
    sys.stdout = original_stdout
    output_file.close()
    print(f"\nGame Output saved to {output_filename}")

if __name__ == "__main__":
    main()