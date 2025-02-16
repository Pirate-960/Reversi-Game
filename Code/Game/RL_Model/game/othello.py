import numpy as np
from enum import Enum

class Player(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

class OthelloGame:
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = Player.BLACK
        self._initialize_board()
        
    def _initialize_board(self):
        """Set up the initial board state"""
        mid = self.size // 2
        self.board[mid-1:mid+1, mid-1:mid+1] = np.array([
            [Player.WHITE.value, Player.BLACK.value],
            [Player.BLACK.value, Player.WHITE.value]
        ])
    
    def get_state(self):
        """Convert game state to neural network input format"""
        state = np.zeros((3, self.size, self.size))
        state[0] = (self.board == self.current_player.value)
        state[1] = (self.board == -self.current_player.value)
        state[2] = (self.board == Player.EMPTY.value)
        return state
    
    def get_valid_moves(self):
        """Return list of valid moves for current player"""
        valid_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self._is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves
    
    def _is_valid_move(self, row, col):
        """Check if move is valid"""
        if self.board[row, col] != Player.EMPTY.value:
            return False
            
        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (-1,-1), (1,-1), (-1,1)]
                     
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc):
                return True
        return False
    
    def _would_flip(self, row, col, dr, dc):
        """Check if placing a piece would flip any opponent pieces"""
        r, c = row + dr, col + dc
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
            
        if self.board[r, c] != -self.current_player.value:
            return False
            
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r, c] == Player.EMPTY.value:
                return False
            if self.board[r, c] == self.current_player.value:
                return True
            r, c = r + dr, c + dc
            
        return False
    
    def make_move(self, move):
        """Apply move to the board"""
        row, col = move
        if not self._is_valid_move(row, col):
            return False
            
        self.board[row, col] = self.current_player.value
        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (-1,-1), (1,-1), (-1,1)]
                     
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc):
                self._flip_pieces(row, col, dr, dc)
                
        self.switch_player()
        return True
    
    def _flip_pieces(self, row, col, dr, dc):
        """Flip opponent pieces after a move"""
        to_flip = []
        r, c = row + dr, col + dc
        
        while 0 <= r < self.size and 0 <= c < self.size:
            if self.board[r, c] == -self.current_player.value:
                to_flip.append((r, c))
            elif self.board[r, c] == self.current_player.value:
                for flip_r, flip_c in to_flip:
                    self.board[flip_r, flip_c] = self.current_player.value
                break
            else:
                break
            r, c = r + dr, c + dc
    
    def switch_player(self):
        """Switch current player"""
        self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
        
        # If current player has no valid moves, switch back
        if not self.get_valid_moves():
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            
            # If other player also has no valid moves, game is over
            if not self.get_valid_moves():
                return False
        return True
    
    def is_game_over(self):
        """Check if game is finished"""
        return (self.board != Player.EMPTY.value).all() or not self.get_valid_moves()
    
    def get_winner_value(self):
        """Get game outcome from black's perspective"""
        black_count = (self.board == Player.BLACK.value).sum()
        white_count = (self.board == Player.WHITE.value).sum()
        
        if black_count > white_count:
            return 1.0
        elif white_count > black_count:
            return -1.0
        return 0.0
    
    def get_score(self):
        """Return current score (black, white)"""
        black_count = (self.board == Player.BLACK.value).sum()
        white_count = (self.board == Player.WHITE.value).sum()
        return black_count, white_count
    
    def display(self):
        """Print current board state"""
        symbols = {
            Player.EMPTY.value: '.',
            Player.BLACK.value: '●',
            Player.WHITE.value: '○'
        }
        print('  ' + ' '.join([str(i) for i in range(self.size)]))
        for i in range(self.size):
            print(f"{i} ", end='')
            for j in range(self.size):
                print(symbols[self.board[i, j]] + ' ', end='')
            print()