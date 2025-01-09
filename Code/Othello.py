import copy
import time
import json
import sys
from datetime import datetime

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # optional: ensures immediate write

    def flush(self):
        for f in self.files:
            f.flush()

class Othello:
    def __init__(self):
        self.board = [["." for _ in range(8)] for _ in range(8)]
        self.board[3][3], self.board[4][4] = "O", "O"
        self.board[3][4], self.board[4][3] = "X", "X"
        self.current_player = "X"
        self.move_history = []
        self.game_log = {
            'game_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now().isoformat(),
            'moves': [],
            'initial_state': self.get_board_state()
        }

    def get_board_state(self):
        """Get current board state with disc positions"""
        state = {
            'board': [row[:] for row in self.board],
            'disc_positions': {
                'X': [],
                'O': []
            }
        }
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == 'X':
                    state['disc_positions']['X'].append(self.numeric_to_algebraic(i, j))
                elif self.board[i][j] == 'O':
                    state['disc_positions']['O'].append(self.numeric_to_algebraic(i, j))
        return state

    def algebraic_to_numeric(self, move):
        """Convert algebraic notation (e.g., 'e3') to numeric coordinates (row, col)"""
        try:
            col = ord(move[0].lower()) - ord('a')
            row = 8 - int(move[1])
            if 0 <= row < 8 and 0 <= col < 8:
                return row, col
            return None
        except (IndexError, ValueError):
            return None

    def numeric_to_algebraic(self, row, col):
        """Convert numeric coordinates to algebraic notation"""
        return f"{chr(col + ord('a'))}{8 - row}"

    def display_board(self):
        print()
        for i in range(8):
            print(f"{8-i} {' '.join(self.board[i])}")
        print("  a b c d e f g h")

    def is_valid_move(self, row, col, player):
        if not (0 <= row < 8 and 0 <= col < 8) or self.board[row][col] != ".":
            return False
            
        opponent = "O" if player == "X" else "X"
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            found_opponent = False

            while 0 <= r < 8 and 0 <= c < 8:
                if self.board[r][c] == opponent:
                    found_opponent = True
                elif self.board[r][c] == player:
                    if found_opponent:
                        return True
                    break
                else:
                    break
                r, c = r + dr, c + dc
        return False

    def get_valid_moves(self, player):
        valid_moves = []
        for r in range(8):
            for c in range(8):
                if self.is_valid_move(r, c, player):
                    valid_moves.append((r, c))
        return valid_moves

    def apply_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False

        opponent = "O" if player == "X" else "X"
        self.board[row][col] = player
        flipped = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            path = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                path.append((r, c))
                r, c = r + dr, c + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                flipped.extend(path)
                for pr, pc in path:
                    self.board[pr][pc] = player

        # Record move details
        flipped_positions = [self.numeric_to_algebraic(r, c) for r, c in flipped]
        move_details = {
            'player': player,
            'move': self.numeric_to_algebraic(row, col),
            'flipped': len(flipped),
            'flipped_positions': flipped_positions
        }
        self.move_history.append(move_details)

        # Log detailed move information
        x_count, o_count = self.count_discs()
        move_log = {
            'move_number': len(self.move_history),
            'timestamp': datetime.now().isoformat(),
            'player': player,
            'move_position': self.numeric_to_algebraic(row, col),
            'flipped_discs': {
                'count': len(flipped),
                'positions': flipped_positions
            },
            'board_state': self.get_board_state(),
            'valid_moves': [self.numeric_to_algebraic(r, c) for r, c in self.get_valid_moves(opponent)],
            'score': {
                'X': x_count,
                'O': o_count
            }
        }
        self.game_log['moves'].append(move_log)
        return True

    def has_valid_move(self, player):
        return bool(self.get_valid_moves(player))

    def switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"

    def count_discs(self):
        x_count = sum(row.count("X") for row in self.board)
        o_count = sum(row.count("O") for row in self.board)
        return x_count, o_count

    def is_game_over(self):
        return not (self.has_valid_move("X") or self.has_valid_move("O"))

    def print_game_status(self):
        x_count, o_count = self.count_discs()
        print(f"\nCurrent score - Black (X): {x_count}, White (O): {o_count}")
        if self.move_history:
            last_move = self.move_history[-1]
            flipped_str = ", ".join(last_move['flipped_positions'])
            print(f"Last move: {last_move['player']} played {last_move['move']}, "
                  f"flipping {last_move['flipped']} disc(s) at: [{flipped_str}]")

    def save_game_log(self, filename=None):
        """Save the game log to a JSON file"""
        if self.game_log['moves']:  # Only save if there were moves made
            self.game_log['end_time'] = datetime.now().isoformat()
            x_count, o_count = self.count_discs()
            self.game_log['final_score'] = {'X': x_count, 'O': o_count}
            self.game_log['winner'] = 'X' if x_count > o_count else 'O' if o_count > x_count else 'Tie'
            
            if filename is None:
                filename = f"othello_game_{self.game_log['game_id']}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.game_log, f, indent=2)
            print(f"\nGame log saved to {filename}")

class OthelloAI:
    def __init__(self, player, depth, heuristic):
        self.player = player
        self.opponent = "O" if player == "X" else "X"
        self.depth = depth
        self.heuristic = heuristic
        self.nodes_evaluated = 0
        self.start_time = 0

    def evaluate(self, game):
        if self.heuristic == 1:
            # h1: Simple disc count difference
            x_count, o_count = game.count_discs()
            return x_count - o_count if self.player == "X" else o_count - x_count
        elif self.heuristic == 2:
            # h2: Enhanced positional strategy
            return self.positional_strategy(game)
        elif self.heuristic == 3:
            # h3: Combined mobility and stability
            return self.mobility_stability(game)

    def positional_strategy(self, game):
        # Position weights
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
                if game.board[i][j] == self.player:
                    score += weights[i][j]
                elif game.board[i][j] == self.opponent:
                    score -= weights[i][j]
        return score

    def mobility_stability(self, game):
        # Mobility: Count of valid moves
        mobility_score = len(game.get_valid_moves(self.player)) - len(game.get_valid_moves(self.opponent))
        
        # Stability: Count stable discs (corners and their adjacent stable discs)
        stability_score = self.count_stable_discs(game)
        
        # Combine scores with weights
        return mobility_score * 10 + stability_score * 30

    def count_stable_discs(self, game):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        stable_count = 0
        
        for corner in corners:
            if game.board[corner[0]][corner[1]] == self.player:
                stable_count += 1
                # Check adjacent positions
                for dr, dc in [(0, 1), (1, 0)] if corner[0] == 0 else [(0, -1), (-1, 0)]:
                    r, c = corner[0] + dr, corner[1] + dc
                    if 0 <= r < 8 and 0 <= c < 8 and game.board[r][c] == self.player:
                        stable_count += 0.5
                        
        return stable_count

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        self.nodes_evaluated += 1
        
        if depth == 0 or game.is_game_over():
            return self.evaluate(game)

        valid_moves = game.get_valid_moves(self.player if maximizing_player else self.opponent)
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
                    break
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
                    break
            return min_eval

    def find_best_move(self, game):
        self.nodes_evaluated = 0
        self.start_time = time.time()
        
        best_move = None
        best_value = float("-inf")
        valid_moves = game.get_valid_moves(self.player)
        
        for r, c in valid_moves:
            new_game = copy.deepcopy(game)
            new_game.apply_move(r, c, self.player)
            move_value = self.minimax(new_game, self.depth - 1, float("-inf"), float("inf"), False)
            if move_value > best_value:
                best_value = move_value
                best_move = (r, c)

        elapsed_time = time.time() - self.start_time
        print(f"\nAI Statistics:")
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Nodes per second: {self.nodes_evaluated / elapsed_time:.0f}")
        
        return best_move

def main():
    # Capture the original stdout
    original_stdout = sys.stdout
    
    print("Welcome to Othello!")
    print("Choose Game Mode:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")
    
    while True:
        try:
            mode = int(input("Enter your choice (1-3): "))
            if mode in [1, 2, 3]:
                break
            print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Initialize variables for filename
    depth = 0
    heuristic1 = 0
    heuristic2 = 0
    
    if mode == 2 or mode == 3:
        while True:
            try:
                depth = int(input("Enter AI search depth (1-8 recommended): "))
                if 1 <= depth <= 10:
                    break
                print("Please enter a reasonable depth (1-8).")
            except ValueError:
                print("Invalid input. Please enter a number.")

        print("\nAvailable heuristics:")
        print("1. Basic disc count")
        print("2. Positional strategy (weighted positions)")
        print("3. Combined mobility and stability")
        
        heuristic1 = int(input("Choose heuristic for AI 1 (1-3): "))
        ai_player1 = OthelloAI("O" if mode == 2 else "X", depth=depth, heuristic=heuristic1)

        if mode == 3:
            heuristic2 = int(input("Choose heuristic for AI 2 (1-3): "))
            ai_player2 = OthelloAI("O", depth=depth, heuristic=heuristic2)

    # Create descriptive filename based on game settings
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if mode == 1:
        output_filename = f"othello_HumanVsHuman_{timestamp}.txt"
    elif mode == 2:
        output_filename = f"othello_HumanVsAI_d{depth}_h{heuristic1}_{timestamp}.txt"
    else:
        output_filename = f"othello_AIvsAI_d{depth}_h{heuristic1}h{heuristic2}_{timestamp}.txt"

    # Open file and set up Tee to write to both console and file
    output_file = open(output_filename, 'w')
    sys.stdout = Tee(sys.stdout, output_file)

    game = Othello()
    game_mode_str = {1: "Human vs Human", 2: "Human vs AI", 3: "AI vs AI"}[mode]
    game.game_log['game_mode'] = game_mode_str
    if mode in [2, 3]:
        game.game_log['ai_settings'] = {
            'AI1': {
                'player': "O" if mode == 2 else "X",
                'depth': depth,
                'heuristic': heuristic1
            }
        }
        if mode == 3:
            game.game_log['ai_settings']['AI2'] = {
                'player': "O",
                'depth': depth,
                'heuristic': heuristic2
            }

    while not game.is_game_over():
        game.display_board()
        game.print_game_status()
        
        if not game.has_valid_move(game.current_player):
            print(f"\n{game.current_player} has no valid moves. Passing...")
            game.switch_player()
            continue

        if mode == 1 or (mode == 2 and game.current_player == "X"):
            print(f"\n{game.current_player}'s turn:")
            moves = game.get_valid_moves(game.current_player)
            print("Available moves:", 
                  [game.numeric_to_algebraic(r, c) for r, c in moves])
            
            while True:
                move = input("Enter your move (e.g., 'e3'): ").strip().lower()
                coords = game.algebraic_to_numeric(move)
                if coords and coords in moves:
                    game.apply_move(coords[0], coords[1], game.current_player)
                    break
                print("Invalid move. Please try again.")
                
        elif mode == 2 and game.current_player == "O":
            print("\nAI's turn (O):")
            move = ai_player1.find_best_move(game)
            if move:
                game.apply_move(move[0], move[1], "O")
                print(f"AI plays: {game.numeric_to_algebraic(move[0], move[1])}")
                
        elif mode == 3:
            print(f"\nAI {game.current_player}'s turn:")
            ai_player = ai_player1 if game.current_player == "X" else ai_player2
            move = ai_player.find_best_move(game)
            if move:
                game.apply_move(move[0], move[1], game.current_player)
                print(f"AI plays: {game.numeric_to_algebraic(move[0], move[1])}")
        
        game.switch_player()

    # Game over
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

    # Save the game log
    game.save_game_log()

    # Restore original stdout and close file
    sys.stdout = original_stdout
    output_file.close()
    print(f"\nGame output has been saved to {output_filename}")

if __name__ == "__main__":
    main()