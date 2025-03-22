import pygame
import random
from smart_player import SmartPlayer

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

COLORS = [
    (0, 255, 255),  # I (cyan)
    (255, 165, 0),  # L (orange)
    (0, 0, 255),  # J (blue)
    (255, 255, 0),  # O (yellow)
    (0, 255, 0),  # S (green)
    (255, 0, 0),  # Z (red)
    (128, 0, 128),  # T (purple)
]

# Tetromino şekilleri
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 0, 0], [1, 1, 1]],  # L
    [[0, 0, 1], [1, 1, 1]],  # J
    [[1, 1], [1, 1]],  # O
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[0, 1, 0], [1, 1, 1]],  # T
]

PIECES = ["I", "O", "T", "L", "J", "S", "Z"]
PIECES_MAP = {piece: (SHAPES[i], COLORS[i]) for i, piece in enumerate(PIECES)}


class Tetris:
    def __init__(self, width=10, height=20, block_size=30):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid = [[0] * width for _ in range(height)]
        self.score = 0
        self.current_piece = None
        self.current_x = 0
        self.current_y = 0
        self.game_over = False
        self.sequence = []
        self.sequence_index = 0
        self.ai = SmartPlayer(self)
        self.autoplay = False

        # Pygame başlatma
        pygame.init()
        self.screen = pygame.display.set_mode(
            (width * block_size + 200, height * block_size)
        )
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
    
    def load_sequence(self, sequence):
        """Load a sequence of pieces"""
        self.sequence = list(sequence)
        self.sequence_index = 0

    def new_piece(self):
        # shape = random.choice(SHAPES)
        # color = random.choice(COLORS)

        if self.sequence_index >= len(self.sequence):
            self.game_over = True
            return
        piece_char = self.sequence[self.sequence_index]
        self.sequence_index += 1

        shape, color = PIECES_MAP[piece_char]

        self.current_piece = {"shape": shape, "color": color}
        self.current_x = self.width // 2 - len(shape[0]) // 2
        self.current_y = 0

        if not self.valid_position(
            self.current_x, self.current_y, self.current_piece["shape"]
        ):
            self.game_over = True

    def valid_position(self, x, y, shape):
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    new_x = x + col
                    new_y = y + row
                    if (
                        new_x < 0
                        or new_x >= self.width
                        or new_y >= self.height
                        or (new_y >= 0 and self.grid[new_y][new_x])
                    ):
                        return False
        return True

    def lock_piece(self):
        for row in range(len(self.current_piece["shape"])):
            for col in range(len(self.current_piece["shape"][row])):
                if self.current_piece["shape"][row][col]:
                    self.grid[self.current_y + row][self.current_x + col] = (
                        self.current_piece["color"]
                    )
        self.clear_lines()
        self.new_piece()

    def clear_lines(self):
        lines_cleared = 0
        for row in range(self.height - 1, -1, -1):
            if all(self.grid[row]):
                del self.grid[row]
                self.grid.insert(0, [0] * self.width)
                lines_cleared += 1
        self.score += lines_cleared * 100

    def move_down(self):
        if self.valid_position(
            self.current_x, self.current_y + 1, self.current_piece["shape"]
        ):
            self.current_y += 1
        else:
            self.lock_piece()

    def move_side(self, dx):
        if self.valid_position(
            self.current_x + dx, self.current_y, self.current_piece["shape"]
        ):
            self.current_x += dx

    def rotate(self):
        rotated = [list(row) for row in zip(*self.current_piece["shape"][::-1])]
        if self.valid_position(self.current_x, self.current_y, rotated):
            self.current_piece["shape"] = rotated

    def draw_grid(self):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(
                    x * self.block_size,
                    y * self.block_size,
                    self.block_size - 1,
                    self.block_size - 1,
                )
                color = self.grid[y][x] if self.grid[y][x] else GRAY
                pygame.draw.rect(self.screen, color, rect)

    def draw_piece(self):
        shape = self.current_piece["shape"]
        color = self.current_piece["color"]
        for y in range(len(shape)):
            for x in range(len(shape[y])):
                if shape[y][x]:
                    rect = pygame.Rect(
                        (self.current_x + x) * self.block_size,
                        (self.current_y + y) * self.block_size,
                        self.block_size - 1,
                        self.block_size - 1,
                    )
                    pygame.draw.rect(self.screen, color, rect)

    def draw_score(self):
        font = pygame.font.SysFont("Arial", 30)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(text, (self.width * self.block_size + 20, 50))

    def handle_input(self, key):
        """Handle input based on key code"""
        if key == pygame.K_LEFT:
            self.move_side(-1)
        elif key == pygame.K_RIGHT:
            self.move_side(1)
        elif key == pygame.K_DOWN:
            self.move_down()
        elif key == pygame.K_UP:
            self.rotate()
        elif key == pygame.K_SPACE:
            while self.valid_position(self.current_x, self.current_y + 1, 
                                    self.current_piece['shape']):
                self.move_down()   
    
    def run(self):
        self.new_piece()
        fall_time = 0
        fall_speed = 500  # ms

        while not self.game_over:
            self.screen.fill(BLACK)
            delta_time = self.clock.tick()
            fall_time += delta_time
            
            if self.autoplay:
                self.ai.update()

            # Olayları işleme
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_side(-1)
                    elif event.key == pygame.K_RIGHT:
                        self.move_side(1)
                    elif event.key == pygame.K_DOWN:
                        self.move_down()
                    elif event.key == pygame.K_UP:
                        self.rotate()

            # Otomatik düşme
            if fall_time >= fall_speed:
                self.move_down()
                fall_time = 0

            # Çizim
            self.draw_grid()
            self.draw_piece()
            self.draw_score()
            pygame.display.update()

        pygame.quit()


def load_sequence(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def select_sequence(sequences):
    print("Available sequences:")
    for i, seq in enumerate(sequences):
        print(f"{i+1}:{seq[:20]}...")
    choice = int(input("Select a sequence (1-100): ")) - 1
    return sequences[choice]


if __name__ == "__main__":

    sequences = load_sequence("tetris_sequences.txt")
    selected_sequence = select_sequence(sequences)
    game = Tetris()
    game.load_sequence(selected_sequence)
    game.autoplay = True
    game.run()
