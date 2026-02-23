# Tetris GA Bot — Adaptive Evolutionary Approach

This repository contains the reference implementation for the paper:

> **Erkalkan, E.** (2026). Heuristic Optimization of a Tetris Bot Using Genetic Algorithms: An Adaptive Evolutionary Approach. Turkish Journal of Mathematics and Computer Science, 18(1), 220–247. https://doi.org/https://doi.org/10.47000/tjmcs.1663275

## Highlights
- Offline GA to learn baseline heuristic weights for a Tetris-playing agent.
- Online (in-game) dynamic reweighting driven by normalized max column height.
- Micro-GA for rapid on-the-fly adaptation (default: population 20, 5 generations).
- Headless simulators for batch experiments and logging.
- Analysis script to compute statistical significance and effect sizes.

## Project Structure
```
tetris/
  tetris.py                     # Pygame UI + core mechanics
  smart_player.py               # Heuristic agent (weights, features, search)
  simulated_tetris.py           # Headless sim for GA-adaptive bot
  simulated_tetris_baseline.py  # Headless sim for fixed-weight baseline
  tetris_sequences.txt          # Pre-generated piece sequences (100 lines)
  baseline_experiment_log.csv   # Results (baseline; 100 runs)
  ga_experiment_log.csv         # Results (adaptive; 100 runs)
  ga_experiment_log_1.csv       # Extra results (adaptive; 100 runs)
  analysis_stats.py             # NEW: significance and effect-size analysis
  requirements.txt              # Dependencies
  LICENSE, CITATION.cff         # Attribution & citation
```

> **Note**: The historic `.git/` directory found in the archive is removed in distributed zips. Keep your own VCS outside shipped archives.

## Installation
We recommend Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1) Run the interactive game (autoplay on a chosen sequence)
```bash
python tetris.py
```
- Choose a sequence from `tetris_sequences.txt` (100 options).
- Toggle `autoplay = True` in `__main__` (already set by default).

### 2) Baseline batch experiment (fixed weights)
```bash
python simulated_tetris_baseline.py
```
Outputs: `baseline_experiment_log.csv` (100 runs by default).

### 3) Adaptive GA experiment
```bash
python simulated_tetris.py
```
- Runs a smaller offline GA for speed, then evaluates adaptive bot.
- Outputs: `ga_experiment_log.csv` (and `ga_experiment_log_1.csv`).

### 4) Statistical analysis & plots
```bash
python analysis_stats.py
```
- Produces `stats_summary.md` (Welch t-tests, effect sizes, CIs).
- Saves histograms and boxplots to `figs/`.

## Reproducibility
- Default seeds are randomized per run; use `PYTHONHASHSEED` and set explicit `random.seed(...)` if you need strict determinism.
- Evaluation count is 100 by default; increase for tighter CIs.

## Known Fixes in This Release
- **APM mislabel**: previous code logged `apm` as total actions; now computed as `actions per minute` in simulators.
- Documentation and citation artifacts added (README, CITATION.cff, LICENSE).
- Added `requirements.txt` and an analysis script with Welch tests and effect sizes.

## License
Released under the MIT License (see `LICENSE`).

## How to Cite
See `CITATION.cff` and the paper citation above.

## Contact
Ercan Erkalkan — <ercan.erkalkan@marmara.edu.tr>
