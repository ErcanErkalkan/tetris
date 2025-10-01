#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

def summarize(name, df):
    return {
        "name": name,
        "n": len(df),
        "lines_mean": df["total_lines_cleared"].mean(),
        "lines_std": df["total_lines_cleared"].std(ddof=1),
        "score_mean": df["max_score"].mean(),
        "score_std": df["max_score"].std(ddof=1),
        "lat_mean": df["decision_latency"].mean(),
        "lat_std": df["decision_latency"].std(ddof=1),
        "dur_mean": df["game_duration"].mean(),
        "dur_std": df["game_duration"].std(ddof=1),
        "apm_mean": df["apm"].mean(),
        "apm_std": df["apm"].std(ddof=1),
    }

def cohen_d(a, b):
    # Pooled SD for independent samples
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp2 = ((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2)
    return (a.mean() - b.mean()) / np.sqrt(sp2)

def welch_t(a, b):
    # Welch's t-test using scipy
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # 95% CI for difference in means using Welch–Satterthwaite approximation
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = a.mean() - b.mean()
    se = np.sqrt(va/na + vb/nb)
    df = (va/na + vb/nb)**2 / ((va**2)/((na**2)*(na-1)) + (vb**2)/((nb**2)*(nb-1)))
    ci = stats.t.interval(0.95, df, loc=diff, scale=se)
    return t, p, df, diff, ci

def main():
    ga = pd.read_csv(BASE / "ga_experiment_log.csv")
    base = pd.read_csv(BASE / "baseline_experiment_log.csv")
    # If extra GA file exists, append
    extra = BASE / "ga_experiment_log_1.csv"
    if extra.exists():
        ga = pd.concat([ga, pd.read_csv(extra)], ignore_index=True)

    # Summary
    s_ga = summarize("GA (adaptive)", ga)
    s_b = summarize("Baseline (fixed)", base)

    # Tests on key metrics
    metrics = ["total_lines_cleared", "max_score", "game_duration", "decision_latency"]
    results = []
    for m in metrics:
        a = ga[m]
        b = base[m]
        t, p, df, diff, ci = welch_t(a, b)
        d = cohen_d(ga[m], base[m])
        results.append({
            "metric": m,
            "mean_ga": a.mean(), "mean_base": b.mean(),
            "diff": diff, "ci_low": ci[0], "ci_high": ci[1],
            "t": t, "df": df, "p": p, "cohen_d": d
        })

    # Write markdown summary
    out = ["# Statistical Summary (Welch t-tests, 95% CI, Cohen's d)",""]
    out.append("## Means (n={} GA, n={} Baseline)".format(len(ga), len(base)))
    out.append(f"- Lines cleared: {s_ga['lines_mean']:.2f} vs {s_b['lines_mean']:.2f}")
    out.append(f"- Duration (min): {s_ga['dur_mean']:.3f} vs {s_b['dur_mean']:.3f}")
    out.append(f"- Score: {s_ga['score_mean']:.2f} vs {s_b['score_mean']:.2f}")
    out.append(f"- Decision latency (ms): {s_ga['lat_mean']:.3f} vs {s_b['lat_mean']:.3f}")
    out.append("")
    out.append("## Welch tests")
    for r in results:
        out.append(f"**{r['metric']}**: diff={r['diff']:.2f} (95% CI {r['ci_low']:.2f}, {r['ci_high']:.2f}), p={r['p']:.3e}, d={r['cohen_d']:.2f}")
    
    # Optional: Static-GA ablation
    static_path = BASE / "static_ga_experiment_log.csv"
    if static_path.exists():
        static_df = pd.read_csv(static_path)
        # report
        s_s = summarize("Static GA (fixed w*)", static_df)
        # Static vs Adaptive
        metrics = ["total_lines_cleared", "max_score", "game_duration", "decision_latency"]
        results_static_vs_adaptive = []
        for m in metrics:
            t, p, dfw, diff, ci = welch_t(static_df[m], ga[m])
            d = cohen_d(static_df[m], ga[m])
            results_static_vs_adaptive.append((m, diff, ci, p, d))

        # Baseline vs Static
        results_base_vs_static = []
        for m in metrics:
            t, p, dfw, diff, ci = welch_t(static_df[m], base[m])
            d = cohen_d(static_df[m], base[m])
            results_base_vs_static.append((m, diff, ci, p, d))

        # Append to markdown
        out.append("")
        out.append("## Ablation (if static_ga_experiment_log.csv present)")
        out.append(f"- Static GA lines cleared: {s_s['lines_mean']:.2f} (vs Adaptive {s_ga['lines_mean']:.2f}, Baseline {s_b['lines_mean']:.2f})")
        out.append("### Static vs Adaptive (Welch)")
        for m, diff, ci, p, d in results_static_vs_adaptive:
            out.append(f"* {m}: diff={diff:.2f} (95% CI {ci[0]:.2f},{ci[1]:.2f}), p={p:.3e}, d={d:.2f}")
        out.append("### Baseline vs Static (Welch)")
        for m, diff, ci, p, d in results_base_vs_static:
            out.append(f"* {m}: diff={diff:.2f} (95% CI {ci[0]:.2f},{ci[1]:.2f}), p={p:.3e}, d={d:.2f}")
    else:
        out.append("")
        out.append("## Ablation")
        out.append("Static-GA log not found (static_ga_experiment_log.csv). Run simulated_tetris_static_ga.py to generate it.")

    (BASE / "stats_summary.md").write_text("\n".join(out), encoding="utf-8")

    # Plots
    figs = BASE / "figs"
    figs.mkdir(exist_ok=True)
    # Histogram and boxplot per key metric
    for m in metrics:
        plt.figure()
        plt.hist(ga[m], alpha=0.6, label="GA (adaptive)")
        plt.hist(base[m], alpha=0.6, label="Baseline")
        plt.xlabel(m); plt.ylabel("count"); plt.legend()
        plt.tight_layout()
        plt.savefig(figs / f"hist_{m}.png", dpi=160)
        plt.close()

        plt.figure()
        plt.boxplot([ga[m], base[m]], labels=["GA","Baseline"])
        plt.ylabel(m)
        plt.tight_layout()
        plt.savefig(figs / f"box_{m}.png", dpi=160)
        plt.close()

    print("Wrote stats_summary.md and figures in ./figs/")

if __name__ == "__main__":
    main()
