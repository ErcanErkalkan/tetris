import time
import csv
import random
import pygame
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
from tetris import Tetris, load_sequence
from smart_player import SmartPlayer

# A subclass of Tetris for headless simulation using the baseline bot with fixed weights.
class SimulatedTetrisBaseline(Tetris):
    def __init__(self, width=10, height=20, block_size=30):
        super().__init__(width, height, block_size)
        self.simulation_mode = True
        self.total_actions = 0
        self.decision_latencies = []
        self.total_lines_cleared = 0
        self.start_time = None
        self.end_time = None
        # Disable display rendering for headless simulation.
        self.screen = None
        self.autoplay = True
        # Baseline bot: create a SmartPlayer instance with fixed heuristic weights.
        self.ai = SmartPlayer(self)
        self.ai.weights = {
            "holes": -5000,
            "rows_with_holes": -700,
            "touching_blocks": 30,
            "max_height": -400,
            "open_sides": -500,
            "next_to_wall": 40,
            "closed_sides": 150,
            "bumpiness": -150,
            "column_filled": 100,
            "lines_cleared": 250,
        }
    
    # Override drawing functions since we don't need to render anything during simulation.
    def draw_grid(self):
        pass

    def draw_piece(self):
        pass

    def draw_score(self):
        pass

    # Override clear_lines to also record the number of cleared lines.
    def clear_lines(self):
        lines_cleared = 0
        for row in range(self.height - 1, -1, -1):
            if all(self.grid[row]):
                del self.grid[row]
                self.grid.insert(0, [0] * self.width)
                lines_cleared += 1
        self.total_lines_cleared += lines_cleared
        self.score += lines_cleared * 100

    # Run a headless simulation of the game and record performance metrics.
    def run_simulation(self):
        self.start_time = time.time()
        self.new_piece()
        while not self.game_over:
            if self.autoplay:
                # Measure decision-making latency.
                decision_start = time.time()
                actions = self.ai.find_best_move(self.current_piece)
                decision_end = time.time()
                latency = (decision_end - decision_start) * 1000  # in milliseconds
                self.decision_latencies.append(latency)
                self.total_actions += len(actions)
                # Execute the actions produced by the AI.
                for action in actions:
                    self.handle_input(action)
            # Gravity: move the piece down if possible.
            if self.valid_position(self.current_x, self.current_y + 1, self.current_piece["shape"]):
                self.current_y += 1
            else:
                self.lock_piece()
        self.end_time = time.time()
        duration_minutes = (self.end_time - self.start_time) / 60.0
        avg_latency = (sum(self.decision_latencies) / len(self.decision_latencies)
                       if self.decision_latencies else 0)
        apm = self.total_actions / duration_minutes if duration_minutes > 0 else 0 / duration_minutes if duration_minutes > 0 else 0
        return {
            "total_lines_cleared": self.total_lines_cleared,
            "max_score": self.score,
            "decision_latency": avg_latency,
            "game_duration": duration_minutes,
            "apm": apm
        }

# Function to run the baseline experiments and log results to a CSV file.
def run_baseline_experiments(num_experiments=100, log_file="baseline_experiment_log.csv"):
    results = []
    sequences = load_sequence("tetris_sequences.txt")
    for i in range(num_experiments):
        print(f"Running simulation {i+1}/{num_experiments}...")
        game = SimulatedTetrisBaseline()
        # Load a random Tetris sequence for each experiment.
        seq = random.choice(sequences)
        game.load_sequence(seq)
        result = game.run_simulation()
        results.append(result)
    
    # Write results to a CSV file.
    keys = ["total_lines_cleared", "max_score", "decision_latency", "game_duration", "apm"]
    with open(log_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    
    # Calculate and print average metrics.
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = sum(r[key] for r in results) / len(results)
    print("\nExperiment completed. Average Metrics:")
    print(f"Total Lines Cleared: {avg_metrics['total_lines_cleared']}")
    print(f"Maximum Score: {avg_metrics['max_score']}")
    print(f"Average Decision Latency (ms): {avg_metrics['decision_latency']}")
    print(f"Average Game Duration (min): {avg_metrics['game_duration']}")
    print(f"Actions Per Minute (APM): {avg_metrics['apm']}")
    
    return results, avg_metrics

if __name__ == "__main__":
    # Initialize pygame display for headless simulation (using dummy video driver if needed).
    pygame.display.init()
    run_baseline_experiments()
