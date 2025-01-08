import pygame
import json
import sys
import time
from pathlib import Path

class OthelloVisualizer:
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 100, 0)
    LIGHT_GREEN = (0, 150, 0)
    GRAY = (128, 128, 128)
    HIGHLIGHT = (255, 255, 0, 128)

    def __init__(self, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Othello Game Visualization")
        
        # Board dimensions
        self.board_size = min(width, height) - 100
        self.cell_size = self.board_size // 8
        self.board_offset_x = (width - self.board_size) // 2
        self.board_offset_y = (height - self.board_size) // 2
        
        # Font initialization
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Animation settings
        self.animation_speed = 10
        self.frame_delay = 50  # milliseconds between moves
        
        # Initialize disc surface with alpha channel for highlights
        self.disc_surface = pygame.Surface((self.cell_size - 4, self.cell_size - 4), pygame.SRCALPHA)

    def load_game(self, filename):
        """Load game data from JSON file"""
        with open(filename, 'r') as f:
            self.game_data = json.load(f)
        self.current_move = -1  # Start before first move
        self.total_moves = len(self.game_data['moves'])

    def draw_board(self):
        """Draw the empty board"""
        # Fill background
        self.screen.fill(self.LIGHT_GREEN)
        
        # Draw board grid
        for i in range(9):
            # Vertical lines
            start_pos = (self.board_offset_x + i * self.cell_size, self.board_offset_y)
            end_pos = (self.board_offset_x + i * self.cell_size, self.board_offset_y + self.board_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos, end_pos, 2)
            
            # Horizontal lines
            start_pos = (self.board_offset_x, self.board_offset_y + i * self.cell_size)
            end_pos = (self.board_offset_x + self.board_size, self.board_offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.BLACK, start_pos, end_pos, 2)

        # Draw coordinate labels
        for i in range(8):
            # Column labels (a-h)
            label = self.small_font.render(chr(ord('a') + i), True, self.BLACK)
            x = self.board_offset_x + i * self.cell_size + self.cell_size//2 - label.get_width()//2
            y = self.board_offset_y + self.board_size + 10
            self.screen.blit(label, (x, y))
            
            # Row labels (1-8)
            label = self.small_font.render(str(8 - i), True, self.BLACK)
            x = self.board_offset_x - 25
            y = self.board_offset_y + i * self.cell_size + self.cell_size//2 - label.get_height()//2
            self.screen.blit(label, (x, y))

    def draw_disc(self, row, col, color, highlight=False):
        """Draw a disc at the specified position"""
        x = self.board_offset_x + col * self.cell_size + 2
        y = self.board_offset_y + row * self.cell_size + 2
        
        # Draw the disc with optional highlight
        if highlight:
            pygame.draw.circle(self.screen, self.HIGHLIGHT,
                             (x + self.cell_size//2, y + self.cell_size//2),
                             self.cell_size//2 - 2)
        pygame.draw.circle(self.screen, color,
                         (x + self.cell_size//2, y + self.cell_size//2),
                         self.cell_size//2 - 4)

    def draw_game_info(self, move_data=None):
        """Draw game information and current move details"""
        if move_data:
            # Draw move number and player
            move_text = f"Move {move_data['move_number']}: {move_data['player']} → {move_data['move_position']}"
            text_surface = self.font.render(move_text, True, self.BLACK)
            self.screen.blit(text_surface, (20, 20))
            
            # Draw score
            score_text = f"Score - Black: {move_data['score']['X']}, White: {move_data['score']['O']}"
            score_surface = self.font.render(score_text, True, self.BLACK)
            self.screen.blit(score_surface, (20, 60))
            
            # Draw flipped discs info
            flipped_text = f"Flipped: {', '.join(move_data['flipped_discs']['positions'])}"
            flipped_surface = self.small_font.render(flipped_text, True, self.BLACK)
            self.screen.blit(flipped_surface, (20, 100))

    def algebraic_to_numeric(self, pos):
        """Convert algebraic notation (e.g., 'e3') to grid coordinates"""
        col = ord(pos[0].lower()) - ord('a')
        row = 8 - int(pos[1])
        return row, col

    def animate_move(self, move_data):
        """Animate the current move and its effects"""
        # Highlight the new move
        row, col = self.algebraic_to_numeric(move_data['move_position'])
        color = self.BLACK if move_data['player'] == 'X' else self.WHITE
        self.draw_disc(row, col, color, highlight=True)
        pygame.display.flip()
        pygame.time.wait(self.frame_delay)
        
        # Animate flipped discs
        for pos in move_data['flipped_discs']['positions']:
            row, col = self.algebraic_to_numeric(pos)
            self.draw_disc(row, col, color, highlight=True)
            pygame.display.flip()
            pygame.time.wait(self.frame_delay)

    def draw_current_state(self):
        """Draw the current state of the board"""
        self.draw_board()
        
        if self.current_move >= 0:
            move_data = self.game_data['moves'][self.current_move]
            board_state = move_data['board_state']
            
            # Draw all discs
            for row in range(8):
                for col in range(8):
                    cell = board_state['board'][row][col]
                    if cell == 'X':
                        self.draw_disc(row, col, self.BLACK)
                    elif cell == 'O':
                        self.draw_disc(row, col, self.WHITE)
            
            # Draw game information
            self.draw_game_info(move_data)
            
            # If this is the latest move, animate it
            if self.current_move == len(self.game_data['moves']) - 1:
                self.animate_move(move_data)
        else:
            # Draw initial state
            initial_state = self.game_data['initial_state']
            for row in range(8):
                for col in range(8):
                    cell = initial_state['board'][row][col]
                    if cell == 'X':
                        self.draw_disc(row, col, self.BLACK)
                    elif cell == 'O':
                        self.draw_disc(row, col, self.WHITE)

    def run(self):
        """Main visualization loop"""
        clock = pygame.time.Clock()
        playing = True
        auto_play = False
        auto_play_delay = 1000  # milliseconds
        last_auto_play_time = 0
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_RIGHT and self.current_move < self.total_moves - 1:
                        self.current_move += 1
                        playing = True
                    elif event.key == pygame.K_LEFT and self.current_move >= 0:
                        self.current_move -= 1
                        playing = True
                    elif event.key == pygame.K_SPACE:
                        auto_play = not auto_play
                    elif event.key == pygame.K_r:  # Reset
                        self.current_move = -1
                        playing = True
                        auto_play = False

            # Handle auto-play
            if auto_play and pygame.time.get_ticks() - last_auto_play_time > auto_play_delay:
                if self.current_move < self.total_moves - 1:
                    self.current_move += 1
                    playing = True
                else:
                    auto_play = False
                last_auto_play_time = pygame.time.get_ticks()

            if playing:
                self.draw_current_state()
                playing = False
                pygame.display.flip()

            clock.tick(60)

def main():
    if len(sys.argv) < 2:
        # python Visualization.py "D:\Github Projects\Reversi Game\othello_game_20250106_134037.json"
        print("Usage: python Visualization.py <game_log.json>")
        return
    
    filename = sys.argv[1]
    if not Path(filename).exists():
        print(f"Error: File '{filename}' not found")
        return
    
    visualizer = OthelloVisualizer()
    visualizer.load_game(filename)
    visualizer.run()

if __name__ == "__main__":
    main()