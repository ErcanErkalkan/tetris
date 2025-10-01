#!/usr/bin/env python3
"""
Static-GA Ablation Runner

Runs 100 headless simulations using the SmartPlayer's optimized weights (w*) 
WITHOUT any in-game dynamic reweighting or online micro-GA. 
Outputs: static_ga_experiment_log.csv
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import csv
import random
import time
import pygame

from tetris import Tetris, load_sequence
from smart_player import SmartPlayer

class SimulatedTetrisStaticGA(Tetris):
    def __init__(self, width=10, height=20, block_size=30):
        super().__init__(width, height, block_size)
        self.simulation_mode = True
        self.total_actions = 0
        self.decision_latencies = []
        self.start_time = None
        self.end_time = None
        # headless
        self.screen = None
        self.autoplay = True
        self.ai = SmartPlayer(self)
        # freeze weights (w* from SmartPlayer initialization)
        self.ai.weights = self.ai.weights.copy()

    def run_simulation(self):
        self.start_time = time.time()
        clock = pygame.time.Clock()

        while not self.game_over:
            if self.autoplay:
                decision_start = time.time()
                actions = self.ai.find_best_move(self.current_piece)
                decision_end = time.time()
                latency = (decision_end - decision_start) * 1000.0
                self.decision_latencies.append(latency)
                self.total_actions += len(actions)
                for key in actions:
                    self.handle_input(key)

            # gravity step (one row)
            self.step()
            # keep simulation fast but stable
            clock.tick(120)

        self.end_time = time.time()
        duration_minutes = (self.end_time - self.start_time) / 60.0 if self.end_time else 0.0
        avg_latency = sum(self.decision_latencies)/len(self.decision_latencies) if self.decision_latencies else 0.0
        apm = self.total_actions / duration_minutes if duration_minutes > 0 else 0.0
        return {
            "total_lines_cleared": self.lines_cleared_total if hasattr(self, "lines_cleared_total") else self.lines_cleared if hasattr(self, "lines_cleared") else 0,
            "max_score": self.score,
            "decision_latency": avg_latency,
            "game_duration": duration_minutes,
            "apm": apm
        }

def run_static_ga_experiments(num_experiments=100, log_path="static_ga_experiment_log.csv"):
    sequences = load_sequence("tetris_sequences.txt")
    keys = ["total_lines_cleared", "max_score", "decision_latency", "game_duration", "apm"]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(num_experiments):
            game = SimulatedTetrisStaticGA()
            seq = random.choice(sequences)
            game.load_sequence(seq)
            result = game.run_simulation()
            writer.writerow(result)
            print(f"[{i+1}/{num_experiments}] lines={result['total_lines_cleared']} score={result['max_score']} dur={result['game_duration']:.3f}m")

    print(f"Wrote {log_path}")

if __name__ == "__main__":
    pygame.init()
    try:
        run_static_ga_experiments(num_experiments=100, log_path="static_ga_experiment_log.csv")
    finally:
        pygame.quit()
