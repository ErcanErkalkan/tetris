import pygame
import random
import math

class SmartPlayer:
    def __init__(self, game):
        self.game = game
        self.action_queue = []

        # Heuristic weights (optimized values)
        # Ağırlıklar
        self.weights = {
            "holes": -4832,  # Deliklere daha yüksek ceza
            "rows_with_holes": -689,
            "touching_blocks": 27,  # Dokunma ödülünü artır
            "max_height": -387,  # Yükseklik cezasını artır
            "open_sides": -521,  # Açık taraflara daha fazla ceza
            "next_to_wall": 40,
            "closed_sides": 183,
            "bumpiness": -150,  # Bumpiness cezasını artır
            "lines_cleared": 252,  # Satır doldurma ödülünü artır
            "column_filled": 100,
        }

        # Tetris bonusları
        self.tetris_bonuses = {
            0: 0,
            1: 100,  # Tek satır temizleme
            2: 8000,  # Çift satır
            3: 20000,  # Üçlü satır
            4: 100000,  # Tetris!
            5: 100000,  # Tetris!
            6: 100000,  # Tetris!
            7: 100000,  # Tetris!
        }

    def calculate_heuristic(self, grid):
        """Calculate heuristic score using all 10 features"""
        features = {
            "holes": self.count_holes(grid),
            "rows_with_holes": self.rows_with_holes(grid),
            "touching_blocks": self.adjacent_touching(grid),
            "max_height": self.max_height(grid),
            "open_sides": self.open_sides(grid),
            "next_to_wall": self.next_to_wall(grid),
            "closed_sides": self.closed_sides(grid),
            "bumpiness": self.calculate_bumpiness(grid),
            "lines_cleared": self.lines_cleared(grid),
            "column_filled": self.column_filled(grid),
        }

        return sum(w * features[k] for k, w in self.weights.items())

    def count_holes(self, grid):
        """Calculate number of holes in the grid"""
        holes = 0
        for x in range(self.game.width):
            found_block = False
            for y in range(self.game.height):
                if grid[y][x]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def rows_with_holes(self, grid):
        """Count rows containing at least one hole"""
        rows = 0
        for y in range(self.game.height):
            has_hole = False
            for x in range(self.game.width):
                if grid[y][x] == 0 and any(grid[yy][x] for yy in range(y)):
                    has_hole = True
                    break
            if has_hole:
                rows += 1
        return rows

    def adjacent_touching(self, grid):
        """Count adjacent filled blocks"""
        touching = 0
        for y in range(self.game.height):
            for x in range(self.game.width):
                if grid[y][x]:
                    # Check all four directions
                    if x > 0 and grid[y][x - 1]:
                        touching += 1  # Left
                    if x < self.game.width - 1 and grid[y][x + 1]:
                        touching += 1  # Right
                    if y > 0 and grid[y - 1][x]:
                        touching += 1  # Up
                    if y < self.game.height - 1 and grid[y + 1][x]:
                        touching += 1  # Down
        return touching

    def max_height(self, grid):
        """Get maximum column height"""
        heights = []
        for x in range(self.game.width):
            for y in range(self.game.height):
                if grid[y][x]:
                    heights.append(self.game.height - y)
                    break
            else:
                heights.append(0)
        return max(heights)

    def open_sides(self, grid):
        """Count exposed cell sides"""
        open_count = 0
        for y in range(self.game.height):
            for x in range(self.game.width):
                if grid[y][x]:
                    # Check all four directions
                    if x == 0 or not grid[y][x - 1]:
                        open_count += 1  # Left
                    if x == self.game.width - 1 or not grid[y][x + 1]:
                        open_count += 1  # Right
                    if y == 0 or not grid[y - 1][x]:
                        open_count += 1  # Up
                    if y == self.game.height - 1 or not grid[y + 1][x]:
                        open_count += 1  # Down
        return open_count

    def next_to_wall(self, grid):
        """Count blocks adjacent to side walls"""
        wall_blocks = 0
        for y in range(self.game.height):
            if grid[y][0]:
                wall_blocks += 1  # Left wall
            if grid[y][self.game.width - 1]:
                wall_blocks += 1  # Right wall
        return wall_blocks

    def closed_sides(self, grid):
        """Count adjacent filled blocks or walls"""
        closed = 0
        for y in range(self.game.height):
            for x in range(self.game.width):
                if grid[y][x]:
                    # Check all four directions
                    if x == 0 or grid[y][x - 1]:
                        closed += 1  # Left
                    if x == self.game.width - 1 or grid[y][x + 1]:
                        closed += 1  # Right
                    if y == 0 or grid[y - 1][x]:
                        closed += 1  # Up
                    if y == self.game.height - 1 or grid[y + 1][x]:
                        closed += 1  # Down
        return closed

    def column_filled(self, grid):
        """Count completely filled columns"""
        filled_cols = 0
        for x in range(self.game.width):
            if all(grid[y][x] for y in range(self.game.height)):
                filled_cols += 1
        return filled_cols

    def calculate_bumpiness(self, grid):
        """Calculate surface unevenness between adjacent columns"""
        heights = self.get_column_heights(grid)
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def get_column_heights(self, grid):
        """Helper method to get column heights"""
        heights = []
        for x in range(self.game.width):
            column_height = 0
            for y in range(self.game.height):
                if grid[y][x]:
                    column_height = self.game.height - y
                    break
            heights.append(column_height)
        return heights

    def lines_cleared(self, grid):
        """Count number of completed lines in current grid"""
        lines = 0
        for y in range(self.game.height):
            if all(grid[y][x] for x in range(self.game.width)):
                lines += 1
        return lines

    def simulate_move(self, piece, rotation, x_pos):
        """Simulate a move and return its heuristic score"""
        # Create copy of grid
        temp_grid = [row.copy() for row in self.game.grid]

        # Simulate piece drop
        y_pos = 0
        while self.game.valid_position(x_pos, y_pos + 1, piece["shape"]):
            y_pos += 1

        # Lock piece in temp grid
        for y in range(len(piece["shape"])):
            for x in range(len(piece["shape"][y])):
                if piece["shape"][y][x]:
                    temp_grid[y_pos + y][x_pos + x] = piece["color"]

        # Clear lines and count cleared
        lines_cleared = 0
        full_lines = []
        for y in range(self.game.height):
            if all(temp_grid[y]):
                full_lines.append(y)

        for y in full_lines:
            del temp_grid[y]
            temp_grid.insert(0, [0] * self.game.width)
            lines_cleared += 1

        # Calculate score with line clearing bonus
        base_score = self.calculate_heuristic(temp_grid)
        return base_score + lines_cleared * self.tetris_bonuses[lines_cleared]

    def find_best_move(self, current_piece):
        best_score = -math.inf
        best_actions = []

        for rotation in range(4):
            piece = current_piece.copy()
            for _ in range(rotation):
                piece["shape"] = [list(row) for row in zip(*piece["shape"][::-1])]

            for x in range(-2, self.game.width + 2):
                if self.game.valid_position(x, 0, piece["shape"]):
                    score = self.simulate_move(piece, rotation, x)
                    if score > best_score:
                        best_score = score
                        best_actions = self.get_actions(current_piece, x, rotation)

        return best_actions

    def get_actions(self, piece, target_x, rotations):
        """Generate action sequence to reach target position"""
        actions = []

        # Rotations
        for _ in range(rotations):
            actions.append(pygame.K_UP)

        # Horizontal movement
        current_x = self.game.current_x
        while current_x != target_x:
            if current_x < target_x:
                actions.append(pygame.K_RIGHT)
                current_x += 1
            else:
                actions.append(pygame.K_LEFT)
                current_x -= 1

        # Hard drop
        actions.append(pygame.K_SPACE)
        return actions

    def update(self):
        """Update AI decisions"""
        if not self.action_queue:
            current_piece = self.game.current_piece.copy()
            self.action_queue = self.find_best_move(current_piece)

        if self.action_queue:
            key = self.action_queue.pop(0)
            self.game.handle_input(key)
