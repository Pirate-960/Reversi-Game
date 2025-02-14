import os
import sys
import unittest

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# import statement
from Code.Game.Othello_v1 import Othello, OthelloAI, Tee

class TestOthello(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        print("\n==========================================================================")
        print("-->>Setting up test fixtures...")
        print("-->>>Creating new Othello game...")
        print("-->>>>Creating new Othello AI player...")
        print("-->>>>>Setup complete.")
        print("==========================================================================\n")
        self.game = Othello()
        self.ai_player = OthelloAI("X", depth=3, heuristic=1)

    def test_initial_board_setup(self):
        """Test if the game board is initialized correctly."""
        # Check board dimensions
        print("Initial Board Setup Test - Board Dimensions\n") 
        for row in self.game.board:
            print(row)
        self.assertEqual(len(self.game.board), 8)
        self.assertEqual(len(self.game.board[0]), 8)
        print("\n")

        # Check initial pieces
        print("Initial Board Setup Test - Initial Pieces\n")
        for row in self.game.board:
            print(row)
        self.assertEqual(self.game.board[3][3], "O")
        self.assertEqual(self.game.board[3][4], "X")
        self.assertEqual(self.game.board[4][3], "X")
        self.assertEqual(self.game.board[4][4], "O")
        
        # Check initial empty spaces
        empty_count = sum(row.count(".") for row in self.game.board)
        self.assertEqual(empty_count, 60)
        print(f"Number of empty spaces: {empty_count}\n")

    def test_coordinate_conversion(self):
        """Test algebraic to numeric coordinate conversion and vice versa."""
        # Test algebraic to numeric
        print("Coordinate Conversion Test - Algebraic to Numeric\n")
        self.assertEqual(self.game.algebraic_to_numeric("e4"), (3, 4))
        print(f"Algebraic 'e4' converted to numeric: {self.game.algebraic_to_numeric('e4')}")
        self.assertEqual(self.game.algebraic_to_numeric("a1"), (0, 0))
        print(f"Algebraic 'a1' converted to numeric: {self.game.algebraic_to_numeric('a1')}")
        self.assertEqual(self.game.algebraic_to_numeric("h8"), (7, 7))
        print(f"Algebraic 'h8' converted to numeric: {self.game.algebraic_to_numeric('h8')}")
        print("\n")
        
        # Test numeric to algebraic
        print("Coordinate Conversion Test - Numeric to Algebraic\n")
        self.assertEqual(self.game.numeric_to_algebraic(3, 4), "e4")
        print(f"Numeric (3, 4) converted to algebraic: {self.game.numeric_to_algebraic(3, 4)}")
        self.assertEqual(self.game.numeric_to_algebraic(0, 0), "a1")
        print(f"Numeric (0, 0) converted to algebraic: {self.game.numeric_to_algebraic(0, 0)}")
        self.assertEqual(self.game.numeric_to_algebraic(7, 7), "h8")
        print(f"Numeric (7, 7) converted to algebraic: {self.game.numeric_to_algebraic(7, 7)}")
        print("\n")

        # Test invalid inputs
        print("Coordinate Conversion Test - Invalid Inputs\n")
        self.assertIsNone(self.game.algebraic_to_numeric("i9"))
        print(f"Invalid algebraic 'i9' converted to numeric: {self.game.algebraic_to_numeric('i9')}")
        self.assertIsNone(self.game.algebraic_to_numeric(""))
        print(f"Invalid algebraic '' converted to numeric: {self.game.algebraic_to_numeric('')}")
        self.assertIsNone(self.game.algebraic_to_numeric("x1"))
        print(f"Invalid algebraic 'x1' converted to numeric: {self.game.algebraic_to_numeric('x1')}")
        print("\n")

    def test_valid_moves(self):
        """Test valid move detection and move application."""
        # Check initial valid moves for black
        valid_moves = self.game.get_valid_moves("X")
        expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]
        self.assertEqual(sorted(valid_moves), sorted(expected_moves))
        print(f"Initial valid moves for 'X': {valid_moves}\n")
        
        # Apply a move and check if it's properly executed
        self.assertTrue(self.game.apply_move(2, 3, "X"))
        self.assertEqual(self.game.board[2][3], "X")
        print(f"Applied move (2, 3) for 'X'. Board state after move:")
        for row in self.game.board:
            print(row)
        print("\n")
        
        # Test invalid moves
        self.assertFalse(self.game.apply_move(0, 0, "X"))  # Invalid position
        self.assertFalse(self.game.apply_move(3, 3, "X"))  # Already occupied
        self.assertFalse(self.game.apply_move(-1, 5, "X"))  # Out of bounds

    def test_piece_flipping(self):
        """Test if pieces are correctly flipped after a move."""
        # Make a move that should flip pieces
        self.game.apply_move(2, 3, "X")
        print(f"Applied move (2, 3) for 'X'. Board state after move:")
        for row in self.game.board:
            print(row)
        print("\n")
        
        # Check if appropriate pieces were flipped
        self.assertEqual(self.game.board[3][3], "X")  # Should be flipped
        print(f"Piece at (3, 3) flipped to 'X'\n")
        
        # Count total pieces after move
        x_count, o_count = self.game.count_discs()
        self.assertEqual(x_count, 4)  # Original 2 + 1 new + 1 flipped
        self.assertEqual(o_count, 1)  # Original 2 - 1 flipped
        print(f"Total 'X' discs: {x_count}, Total 'O' discs: {o_count}\n")

    def test_game_over_conditions(self):
        """Test game over detection."""
        # New game should not be over
        self.assertFalse(self.game.is_game_over())
        print("New game is not over.\n")
        
        # Fill board except for one square
        for i in range(8):
            for j in range(8):
                if self.game.board[i][j] == ".":
                    self.game.board[i][j] = "X"
        
        # Game should be over when no valid moves remain
        self.assertTrue(self.game.is_game_over())
        print("Game is over after filling the board.\n")

    def test_score_counting(self):
        """Test disc counting functionality."""
        # Check initial counts
        x_count, o_count = self.game.count_discs()
        self.assertEqual(x_count, 2)
        self.assertEqual(o_count, 2)
        print(f"Initial counts - 'X': {x_count}, 'O': {o_count}\n")
        
        # Make a move and check updated counts
        self.game.apply_move(2, 3, "X")
        x_count, o_count = self.game.count_discs()
        self.assertEqual(x_count, 4)
        self.assertEqual(o_count, 1)
        print(f"Counts after move (2, 3) - 'X': {x_count}, 'O': {o_count}\n")

    def test_ai_evaluation(self):
        """Test AI evaluation functions."""
        # Test basic disc count heuristic
        ai_basic = OthelloAI("X", depth=3, heuristic=1)
        initial_eval = ai_basic.evaluate(self.game)
        self.assertEqual(initial_eval, 0)  # Should be 0 for initial position
        print(f"Basic heuristic evaluation: {initial_eval}\n")
        
        # Test positional strategy heuristic
        ai_positional = OthelloAI("X", depth=3, heuristic=2)
        pos_eval = ai_positional.evaluate(self.game)
        self.assertIsInstance(pos_eval, (int, float))
        print(f"Positional heuristic evaluation: {pos_eval}\n")
        
        # Test mobility/stability heuristic
        ai_mobility = OthelloAI("X", depth=3, heuristic=3)
        mob_eval = ai_mobility.mobility_stability(self.game)
        self.assertIsInstance(mob_eval, (int, float))
        print(f"Mobility/Stability heuristic evaluation: {mob_eval}\n")

    def test_ai_move_generation(self):
        """Test AI move generation."""
        # Ensure AI generates valid moves
        move = self.ai_player.find_best_move(self.game)
        self.assertIsNotNone(move)
        self.assertTrue(self.game.is_valid_move(move[0], move[1], "X"))
        print(f"AI generated move: {move}\n")
        
        # Test AI with different depths
        shallow_ai = OthelloAI("X", depth=1, heuristic=1)
        deep_ai = OthelloAI("X", depth=6, heuristic=1)
        
        shallow_move = shallow_ai.find_best_move(self.game)
        deep_move = deep_ai.find_best_move(self.game)
        
        self.assertTrue(self.game.is_valid_move(shallow_move[0], shallow_move[1], "X"))
        self.assertTrue(self.game.is_valid_move(deep_move[0], deep_move[1], "X"))
        print("\nAI move generation test results:\n")
        print(f"Shallow AI move: {shallow_move}")
        print(f"Deep AI move: {deep_move}\n")

    def test_game_log(self):
        """Test game logging functionality."""
        # Check initial log structure
        self.assertIn('game_id', self.game.game_log)
        self.assertIn('start_time', self.game.game_log)
        self.assertIn('moves', self.game.game_log)
        self.assertIn('initial_state', self.game.game_log)
        print("Initial game log structure is correct.\n")
        
        # Make a move and check log update
        self.game.apply_move(2, 3, "X")
        self.assertEqual(len(self.game.game_log['moves']), 1)
        print(f"Game log updated with move: {self.game.game_log['moves'][-1]}\n")
        
        # Check move log details
        last_move = self.game.game_log['moves'][-1]
        self.assertIn('move_number', last_move)
        self.assertIn('timestamp', last_move)
        self.assertIn('player', last_move)
        self.assertIn('move_position', last_move)
        self.assertIn('flipped_discs', last_move)
        self.assertIn('board_state', last_move)
        self.assertIn('valid_moves', last_move)
        self.assertIn('score', last_move)
        print("Move log details are correct.\n")

    def test_player_switching(self):
        """Test player turn switching."""
        self.assertEqual(self.game.current_player, "X")
        print(f"Current player before switch: {self.game.current_player}")
        self.game.switch_player()
        self.assertEqual(self.game.current_player, "O")
        print(f"Current player after first switch: {self.game.current_player}")
        self.game.switch_player()
        self.assertEqual(self.game.current_player, "X")
        print(f"Current player after second switch: {self.game.current_player}\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)