#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alt_heur_runner_project.py
--------------------------
Bu sürüm, projenizdeki `simulated_tetris.py` içindeki SimulatedTetrisGA sınıfını
**doğrudan** kullanır ve tek amaç olarak "ortalama lines cleared" değerini maksimize eder.
DE ve PSO içerir (opsiyonel hafif CMA-ES eklenebilir).

Kullanım (tetris dizininde, yani simulated_tetris.py ile aynı klasörde çalıştırın):
  python alt_heur_runner_project.py --algo de  --iters 50 --pop 40 --games 30
  python alt_heur_runner_project.py --algo pso --iters 60 --pop 40 --games 30

Çıktılar:
  ./alt_heur_results.csv
  ./alt_heur_best.json
Terminal çıktısında LaTeX tablosuna koyacağınız "Lines Cleared (avg)" değeri yazdırılır.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple

import numpy as np

# -------------------------------------------------------------
# Ortam kur: bu dosyanın klasörünü sys.path'e ekle
# -------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in os.sys.path:
    os.sys.path.insert(0, HERE)

# Proje modüllerini içeri al
sim_mod = importlib.import_module("simulated_tetris")   # aynı klasörde olmalı
# simulated_tetris modülünde "from tetris import Tetris, load_sequence" yapılmış durumda
# bu yüzden load_sequence fonksiyonuna sim_mod.load_sequence ile erişilebilir.
load_sequence = getattr(sim_mod, "load_sequence")

# Baz ağırlıkları ve özellik isimleri (çekirdek sıra)
if hasattr(sim_mod, "base_weights"):
    BASE_WEIGHTS = sim_mod.base_weights
else:
    # Yedek: makaledeki isimler
    BASE_WEIGHTS = {
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
FEATURES = list(BASE_WEIGHTS.keys())  # dict insertion order korunur

# Varsayılan kutu sınırları (gerekirse --bounds ile değiştirilebilir)
DEFAULT_LOWER = np.array([-10000, -2000,    0, -1000, -1500,    0,    0,  -500,    0,   0 ], dtype=float)
DEFAULT_UPPER = np.array([  -1000,  -100,  200,   -50,   -50,  200,  400,     0,  300, 500], dtype=float)

# -------------------------------------------------------------
# Yardımcılar
# -------------------------------------------------------------
def project_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(upper, np.maximum(lower, x))

def seed_all(seed: Optional[int] = None):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

@dataclass
class RunResult:
    best_x: np.ndarray
    best_f: float
    history: List[float]

# -------------------------------------------------------------
# Değerlendirme (objective): ortalama lines cleared
# -------------------------------------------------------------
def vector_to_chromosome(vec: np.ndarray) -> dict:
    return {FEATURES[i]: float(vec[i]) for i in range(len(FEATURES))}

def eval_lines_cleared(weights_vec: np.ndarray, n_games: int = 30, seed: Optional[int] = None) -> float:
    """
    SimulatedTetrisGA kullanarak belirtilen ağırlık vektörü için "ortalama lines cleared" döndürür.
    """
    rng = np.random.default_rng(seed)
    chrom = vector_to_chromosome(weights_vec)

    # tetris_sequences.txt yolu (aynı klasörde)
    seq_path = os.path.join(HERE, "tetris_sequences.txt")
    if not os.path.isfile(seq_path):
        # üst dizinde olabilir
        seq_path = os.path.join(os.path.dirname(HERE), "tetris_sequences.txt")
    sequences = load_sequence(seq_path if os.path.isabs(seq_path) else "tetris_sequences.txt")

    total_lines = 0.0
    for _ in range(n_games):
        game = sim_mod.SimulatedTetrisGA(chrom)  # GA sürümü, dinamik ağırlıkları kullanıyor
        game.ai.weights = chrom.copy()
        # rastgele bir dizi seç ve yükle
        seq = sequences[int(rng.integers(len(sequences)))]
        game.load_sequence(seq)
        res = game.run_simulation()
        total_lines += float(res.get("total_lines_cleared", 0.0))
    return total_lines / max(1, n_games)

# -------------------------------------------------------------
# DE ve PSO
# -------------------------------------------------------------
def de_optimize(objective: Callable[[np.ndarray, int, Optional[int]], float],
                lower: np.ndarray, upper: np.ndarray,
                pop_size: int = 40, iters: int = 50, n_games: int = 30,
                F: float = 0.5, CR: float = 0.9, seed: Optional[int] = None) -> RunResult:
    dim = lower.shape[0]
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.array([objective(pop[i], n_games, int(rng.integers(1e9))) for i in range(pop_size)], dtype=float)

    best_idx = int(np.argmax(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    history = [best_f]

    for _ in range(iters):
        for i in range(pop_size):
            idxs = list(range(pop_size)); idxs.remove(i)
            a, b, c = rng.choice(idxs, size=3, replace=False)
            v = pop[a] + F * (pop[b] - pop[c])
            v = project_bounds(v, lower, upper)
            j_rand = rng.integers(0, dim)
            u = np.array([v[j] if (rng.random() < CR or j == j_rand) else pop[i, j] for j in range(dim)], dtype=float)
            u = project_bounds(u, lower, upper)
            fu = objective(u, n_games, int(rng.integers(1e9)))
            if fu >= fitness[i]:
                pop[i] = u; fitness[i] = fu
                if fu >= best_f:
                    best_f, best_x = float(fu), u.copy()
        history.append(best_f)

    return RunResult(best_x=best_x, best_f=best_f, history=history)

def pso_optimize(objective: Callable[[np.ndarray, int, Optional[int]], float],
                 lower: np.ndarray, upper: np.ndarray,
                 pop_size: int = 40, iters: int = 60, n_games: int = 30,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 vmax_frac: float = 0.2, seed: Optional[int] = None) -> RunResult:
    dim = lower.shape[0]; rng = np.random.default_rng(seed)
    x = rng.uniform(lower, upper, size=(pop_size, dim))
    v = np.zeros_like(x); vmax = vmax_frac * (upper - lower)

    pbest = x.copy()
    pbest_f = np.array([objective(x[i], n_games, int(rng.integers(1e9))) for i in range(pop_size)], dtype=float)
    g_idx = int(np.argmax(pbest_f)); gbest = pbest[g_idx].copy(); gbest_f = float(pbest_f[g_idx])
    history = [gbest_f]

    for _ in range(iters):
        r1 = rng.random(size=(pop_size, dim)); r2 = rng.random(size=(pop_size, dim))
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        v = np.clip(v, -vmax, vmax)
        x = project_bounds(x + v, lower, upper)

        fvals = np.array([objective(x[i], n_games, int(rng.integers(1e9))) for i in range(pop_size)], dtype=float)
        improved = fvals >= pbest_f
        pbest[improved] = x[improved]; pbest_f[improved] = fvals[improved]
        g_idx = int(np.argmax(pbest_f))
        if pbest_f[g_idx] >= gbest_f:
            gbest_f, gbest = float(pbest_f[g_idx]), pbest[g_idx].copy()
        history.append(gbest_f)

    return RunResult(best_x=gbest, best_f=gbest_f, history=history)

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DE/PSO ile Tetris ağırlık optimizasyonu (avg lines cleared).")
    p.add_argument("--algo", choices=["de","pso"], default="de", help="Algoritma")
    p.add_argument("--pop", type=int, default=40, help="Popülasyon/Parçacık sayısı")
    p.add_argument("--iters", type=int, default=50, help="İterasyon sayısı")
    p.add_argument("--games", type=int, default=30, help="Değerlendirmede oyun sayısı (ortalama)")
    p.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    p.add_argument("--bounds", type=str, default="", help="Alt/üst sınırlar için JSON (opsiyonel)")
    p.add_argument("--outdir", type=str, default=".", help="Çıktı dizini")
    return p.parse_args()

def load_bounds(bounds_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    if bounds_path and os.path.isfile(bounds_path):
        with open(bounds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lower = np.array(data.get("lower", DEFAULT_LOWER).copy(), dtype=float)
        upper = np.array(data.get("upper", DEFAULT_UPPER).copy(), dtype=float)
        assert lower.shape == (len(FEATURES),) and upper.shape == (len(FEATURES),)
        return lower, upper
    return DEFAULT_LOWER.copy(), DEFAULT_UPPER.copy()

def main():
    args = parse_args()
    seed_all(args.seed)
    lower, upper = load_bounds(args.bounds)

    # Eğer sınır boyutu FEATURES ile uyuşmuyorsa uyarı yazdır
    if lower.shape[0] != len(FEATURES) or upper.shape[0] != len(FEATURES):
        print("[WARN] bounds dimension != feature count; using defaults.")
        lower, upper = DEFAULT_LOWER.copy(), DEFAULT_UPPER.copy()

    t0 = time.time()
    if args.algo == "de":
        res = de_optimize(eval_lines_cleared, lower, upper, pop_size=args.pop, iters=args.iters, n_games=args.games, seed=args.seed)
        algo_name = "DE"
    else:
        res = pso_optimize(eval_lines_cleared, lower, upper, pop_size=args.pop, iters=args.iters, n_games=args.games, seed=args.seed)
        algo_name = "PSO"
    t1 = time.time()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "alt_heur_results.csv")
    json_path = os.path.join(args.outdir, "alt_heur_best.json")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("algo,iters,pop,games,best_f,elapsed_s\n")
        f.write(f"{algo_name},{args.iters},{args.pop},{args.games},{res.best_f:.4f},{(t1-t0):.3f}\n")

    out = {
        "algo": algo_name,
        "iters": args.iters,
        "pop": args.pop,
        "games": args.games,
        "best_f": res.best_f,
        "best_weights": res.best_x.tolist(),
        "history": res.history,
        "elapsed_s": (t1 - t0),
        "features": FEATURES,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] {algo_name} bitti. Lines Cleared (avg) ~ {res.best_f:.3f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print("--- LaTeX tablo için ---")
    print(f"Lines Cleared (avg): {res.best_f:.2f}")

if __name__ == "__main__":
    main()
