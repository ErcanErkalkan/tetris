
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alt_heur_runner_project_fair.py
-------------------------------
SimulatedTetrisGA kullanarak DE ve PSO ile "avg lines cleared" optimizasyonu.
Adil kıyas için iki yol:
  (A) --profile {rev2_small, ga_heavy, medium}
  (B) --budget N ile toplam fitness çağrısı eşitleme (iters = ceil(budget/pop))

Örnekler:
  # Reviewer-2 küçük profil (eşit bütçe): 2000 değerlendirme, oyun başına 30 ortalama
  python alt_heur_runner_project_fair.py --algo de  --profile rev2_small
  python alt_heur_runner_project_fair.py --algo pso --profile rev2_small

  # GA-heavy (makaledeki offline GA ile aynı büyüklük): 10,000 değerlendirme, 100 oyun ortalaması
  python alt_heur_runner_project_fair.py --algo de  --profile ga_heavy
  python alt_heur_runner_project_fair.py --algo pso --profile ga_heavy

  # Eşit toplam değerlendirme: 3000 fitness çağrısı, 50 oyun ortalaması
  python alt_heur_runner_project_fair.py --algo de  --budget 3000 --games 50 --pop 50
  python alt_heur_runner_project_fair.py --algo pso --budget 3000 --games 50 --pop 50

Çıktılar: alt_heur_results.csv, alt_heur_best.json
Terminal: LaTeX tablo için "Lines Cleared (avg)"
"""
from __future__ import annotations
import argparse, importlib, json, math, os, random, time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in os.sys.path:
    os.sys.path.insert(0, HERE)

sim_mod = importlib.import_module("simulated_tetris")
load_sequence = getattr(sim_mod, "load_sequence")

if hasattr(sim_mod, "base_weights"):
    BASE_WEIGHTS = sim_mod.base_weights
else:
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
FEATURES = list(BASE_WEIGHTS.keys())

DEFAULT_LOWER = np.array([-10000, -2000,    0, -1000, -1500,    0,    0,  -500,    0,   0 ], dtype=float)
DEFAULT_UPPER = np.array([  -1000,  -100,  200,   -50,   -50,  200,  400,     0,  300, 500], dtype=float)

def project_bounds(x, lower, upper): return np.minimum(upper, np.maximum(lower, x))
def seed_all(seed: Optional[int]): 
    if seed is None: return
    random.seed(seed); np.random.seed(seed % (2**32 - 1))

@dataclass
class RunResult:
    best_x: np.ndarray
    best_f: float
    history: List[float]

def vector_to_chromosome(vec: np.ndarray) -> dict:
    return {FEATURES[i]: float(vec[i]) for i in range(len(FEATURES))}

def eval_lines_cleared(weights_vec: np.ndarray, n_games: int = 30, seed: Optional[int] = None) -> float:
    rng = np.random.default_rng(seed)
    chrom = vector_to_chromosome(weights_vec)
    seq_path = os.path.join(HERE, "tetris_sequences.txt")
    if not os.path.isfile(seq_path):
        seq_path = os.path.join(os.path.dirname(HERE), "tetris_sequences.txt")
    sequences = load_sequence(seq_path if os.path.isabs(seq_path) else "tetris_sequences.txt")
    total_lines = 0.0
    for _ in range(n_games):
        game = sim_mod.SimulatedTetrisGA(chrom)
        game.ai.weights = chrom.copy()
        seq = sequences[int(rng.integers(len(sequences)))]
        game.load_sequence(seq)
        res = game.run_simulation()
        total_lines += float(res.get("total_lines_cleared", 0.0))
    return total_lines / max(1, n_games)

def de_optimize(objective: Callable[[np.ndarray, int, Optional[int]], float],
                lower, upper, pop_size=40, iters=50, n_games=30,
                F=0.5, CR=0.9, seed: Optional[int]=None) -> RunResult:
    dim = lower.shape[0]; rng = np.random.default_rng(seed)
    pop = rng.uniform(lower, upper, size=(pop_size, dim))
    fitness = np.array([objective(pop[i], n_games, int(rng.integers(1e9))) for i in range(pop_size)], dtype=float)
    best_idx = int(np.argmax(fitness)); best_x = pop[best_idx].copy(); best_f = float(fitness[best_idx])
    history = [best_f]
    for _ in range(iters):
        for i in range(pop_size):
            idxs = list(range(pop_size)); idxs.remove(i)
            a,b,c = rng.choice(idxs, size=3, replace=False)
            v = project_bounds(pop[a] + F*(pop[b]-pop[c]), lower, upper)
            j_rand = rng.integers(0, dim)
            u = np.array([v[j] if (rng.random()<CR or j==j_rand) else pop[i,j] for j in range(dim)], dtype=float)
            u = project_bounds(u, lower, upper)
            fu = objective(u, n_games, int(rng.integers(1e9)))
            if fu >= fitness[i]:
                pop[i] = u; fitness[i] = fu
                if fu >= best_f: best_f, best_x = float(fu), u.copy()
        history.append(best_f)
    return RunResult(best_x=best_x, best_f=best_f, history=history)

def pso_optimize(objective: Callable[[np.ndarray, int, Optional[int]], float],
                 lower, upper, pop_size=40, iters=60, n_games=30,
                 w=0.7, c1=1.5, c2=1.5, vmax_frac=0.2, seed: Optional[int]=None) -> RunResult:
    dim = lower.shape[0]; rng = np.random.default_rng(seed)
    x = rng.uniform(lower, upper, size=(pop_size, dim)); v = np.zeros_like(x)
    vmax = vmax_frac * (upper - lower)
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
        if pbest_f[g_idx] >= gbest_f: gbest_f, gbest = float(pbest_f[g_idx]), pbest[g_idx].copy()
        history.append(gbest_f)
    return RunResult(best_x=gbest, best_f=gbest_f, history=history)

def parse_args():
    p = argparse.ArgumentParser(description="DE/PSO adil kıyas koşullarıyla (budget/profile)")
    p.add_argument("--algo", choices=["de","pso"], default="de")
    p.add_argument("--profile", choices=["", "rev2_small", "ga_heavy", "medium"], default="",
                   help="Ön-tanımlı adil kıyas profilleri")
    p.add_argument("--budget", type=int, default=0, help="Toplam fitness çağrısı (iters=ceil(budget/pop))")
    p.add_argument("--pop", type=int, default=40, help="Popülasyon/Parçacık sayısı")
    p.add_argument("--iters", type=int, default=0, help="Profil/bütçe yerine manuel iterasyon")
    p.add_argument("--games", type=int, default=30, help="Her fitness için kaç oyun ortalaması")
    p.add_argument("--seed", type=int, default=2025, help="Rastgelelik tohumu (adil kıyas için sabit tutun)")
    p.add_argument("--bounds", type=str, default="", help="Alt/üst sınırlar JSON (opsiyonel)")
    p.add_argument("--outdir", type=str, default=".", help="Çıktı dizini")
    return p.parse_args()

def load_bounds(bounds_path: Optional[str]):
    if bounds_path and os.path.isfile(bounds_path):
        with open(bounds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lower = np.array(data.get("lower", DEFAULT_LOWER).copy(), dtype=float)
        upper = np.array(data.get("upper", DEFAULT_UPPER).copy(), dtype=float)
        assert lower.shape == (len(FEATURES),) and upper.shape == (len(FEATURES),)
        return lower, upper
    return DEFAULT_LOWER.copy(), DEFAULT_UPPER.copy()

def resolve_schedule(args):
    # Profil -> (pop, iters, games)
    if args.profile == "rev2_small":
        pop, iters, games = 40, 50, 30   # 2000 fitness çağrısı, 30 oyun ortalaması
    elif args.profile == "medium":
        pop, iters, games = 50, 60, 50   # 3000 çağrı, 50 oyun
    elif args.profile == "ga_heavy":
        pop, iters, games = 100, 100, 100  # 10,000 çağrı, 100 oyun (offline GA ile eşit yük)
    else:
        pop, iters, games = args.pop, args.iters if args.iters>0 else 50, args.games
        if args.budget > 0 and pop > 0:
            iters = (args.budget + pop - 1)//pop
    return pop, iters, games

def main():
    args = parse_args()
    seed_all(args.seed)
    lower, upper = load_bounds(args.bounds)
    pop, iters, games = resolve_schedule(args)

    t0 = time.time()
    if args.algo == "de":
        res = de_optimize(eval_lines_cleared, lower, upper, pop_size=pop, iters=iters, n_games=games, seed=args.seed)
        algo_name = "DE"
    else:
        res = pso_optimize(eval_lines_cleared, lower, upper, pop_size=pop, iters=iters, n_games=games, seed=args.seed)
        algo_name = "PSO"
    t1 = time.time()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "alt_heur_results.csv")
    json_path = os.path.join(args.outdir, "alt_heur_best.json")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("algo,profile,iters,pop,games,best_f,elapsed_s\n")
        f.write(f"{algo_name},{args.profile or 'custom'},{iters},{pop},{games},{res.best_f:.4f},{(t1-t0):.3f}\n")

    out = {
        "algo": algo_name, "profile": args.profile or "custom",
        "iters": iters, "pop": pop, "games": games,
        "best_f": res.best_f, "best_weights": res.best_x.tolist(),
        "history": res.history, "elapsed_s": (t1 - t0), "features": FEATURES,
        "seed": args.seed
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] {algo_name} ({args.profile or 'custom'}) bitti.")
    print(f"Toplam fitness çağrısı ~= {iters*pop} ; Oyun ortalaması / çağrı = {games}")
    print(f"Lines Cleared (avg): {res.best_f:.2f}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

if __name__ == "__main__":
    main()
