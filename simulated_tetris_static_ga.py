#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static-GA Ablation Runner (based on your dynamic GA scaffold)

- Weights are fixed to best_weights (no in-game adjust_weights)
- Plan is computed ONCE per piece (action queue), then applied step-by-step
- Gravity/lock order mirrors baseline
- Robust lines-cleared counting via overridden clear_lines()
- Output CSV: static_ga_experiment_log.csv  (headers match your GA logs)
"""

import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # headless pygame

import time
import csv
import random
import pygame

from tetris import Tetris, load_sequence
from smart_player import SmartPlayer


# ------------------------------
# Helper: robust lines extractor
# ------------------------------
def _get_lines_cleared(game) -> int:
    """
    Try multiple names; fall back to our own total_lines_cleared.
    """
    for name in [
        "total_lines_cleared",
        "lines_cleared_total",
        "lines_cleared",
        "cleared_lines",
        "line_count",
        "lines",
    ]:
        if hasattr(game, name):
            try:
                return int(getattr(game, name))
            except Exception:
                pass
    return 0


# -------------------------------------------------
# Static (no adaptation): SimulatedTetrisStaticGA
# -------------------------------------------------
class SimulatedTetrisStaticGA(Tetris):
    def __init__(self, best_weights, width=10, height=20, block_size=30):
        super().__init__(width, height, block_size)
        self.simulation_mode = True
        self.total_actions = 0
        self.decision_latencies = []
        self.total_lines_cleared = 0  # our own counter (for robust logging)
        self.start_time = None
        self.end_time = None

        # Headless
        self.screen = None
        self.autoplay = True

        # AI + fixed weights (w*)
        self.ai = SmartPlayer(self)
        self.best_weights = best_weights.copy()
        self.ai.weights = self.best_weights.copy()

        # Action queue: plan once per piece
        self.action_queue = []

    # --- visual stubs (headless) ---
    def draw_grid(self):  pass
    def draw_piece(self): pass
    def draw_score(self): pass

    # --- robust line clearing (override) ---
    def clear_lines(self):
        lines_cleared = 0
        for row in range(self.height - 1, -1, -1):
            if all(self.grid[row]):
                del self.grid[row]
                self.grid.insert(0, [0] * self.width)
                lines_cleared += 1
        self.total_lines_cleared += lines_cleared
        # keep same scoring scheme as your dynamic code
        self.score += lines_cleared * 100

    # --- planning: compute once per piece ---
    def _plan_if_needed(self):
        if self.autoplay and self.current_piece and not self.action_queue:
            t0 = time.time()
            plan = self.ai.find_best_move(self.current_piece) or []
            t1 = time.time()
            self.decision_latencies.append((t1 - t0) * 1000.0)
            # enqueue plan
            self.action_queue = list(plan)

    # --- apply one action per tick (prevents APM explosion) ---
    def _apply_one_action(self):
        if self.action_queue:
            key = self.action_queue.pop(0)
            self.total_actions += 1
            self.handle_input(key)

    # --- main loop ---
    def run_simulation(self):
        self.start_time = time.time()
        clock = pygame.time.Clock()

        # ensure first piece
        self.new_piece()

        while not self.game_over:
            # 1) plan (only when queue empty/new piece)
            self._plan_if_needed()
            # 2) apply single action from queue
            self._apply_one_action()

            # 3) gravity: one row, else lock
            if self.valid_position(self.current_x, self.current_y + 1, self.current_piece["shape"]):
                self.current_y += 1
            else:
                self.lock_piece()
                # after locking, engine typically spawns next piece,
                # so queue will be empty => new plan next loop

            # fast & stable loop
            clock.tick(120)

        self.end_time = time.time()

        # metrics
        duration_minutes = (self.end_time - self.start_time) / 60.0 if self.end_time else 0.0
        avg_latency = (sum(self.decision_latencies) / len(self.decision_latencies)) if self.decision_latencies else 0.0
        apm = (self.total_actions / duration_minutes) if duration_minutes > 0 else 0.0

        return {
            "total_lines_cleared": _get_lines_cleared(self) or self.total_lines_cleared,
            "max_score": self.score,
            "decision_latency": avg_latency,
            "game_duration": duration_minutes,
            "apm": apm,
        }


# ---------------------------------------------
# Runner: 100 sims → static_ga_experiment_log.csv
# ---------------------------------------------
def run_static_ga_experiments(best_weights, num_experiments=100, log_file="static_ga_experiment_log.csv"):
    sequences = load_sequence("tetris_sequences.txt")
    results = []
    for i in range(num_experiments):
        game = SimulatedTetrisStaticGA(best_weights)
        seq = random.choice(sequences)
        game.load_sequence(seq)
        res = game.run_simulation()
        results.append(res)
        print(f"[{i+1:03d}/{num_experiments}] "
              f"lines={res['total_lines_cleared']} "
              f"score={res['max_score']} "
              f"dur={res['game_duration']:.3f}m "
              f"apm={res['apm']:.0f} "
              f"lat={res['decision_latency']:.2f}ms")

    # write CSV (same headers as your GA logs)
    keys = ["total_lines_cleared", "max_score", "decision_latency", "game_duration", "apm"]
    with open(log_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    print(f"\nWrote {log_file}")
    return results


# -------------------------------------------------
# Optional: quick entry (use your own best_weights)
# -------------------------------------------------
if __name__ == "__main__":
    pygame.init()
    try:
        # If you already have best_weights from your GA optimizer, put them here:
        best_weights = {
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
        # ↑ bunu GA ile bulduğun w* ile değiştirmen önerilir.

        run_static_ga_experiments(best_weights, num_experiments=100,
                                  log_file="static_ga_experiment_log.csv")
    finally:
        pygame.quit()
