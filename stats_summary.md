# Statistical Summary (Welch t-tests, 95% CI, Cohen's d)

## Means (n=200 GA, n=100 Baseline)
- Lines cleared: 79.06 vs 56.48
- Duration (min): 0.027 vs 0.020
- Score: 7905.50 vs 5648.00
- Decision latency (ms): 7.130 vs 7.054

## Welch tests
**total_lines_cleared**: diff=22.58 (95% CI 9.70, 35.45), p=6.536e-04, d=0.40
**max_score**: diff=2257.50 (95% CI 970.43, 3544.57), p=6.536e-04, d=0.40
**game_duration**: diff=0.01 (95% CI 0.00, 0.01), p=3.560e-04, d=0.42
**decision_latency**: diff=0.08 (95% CI 0.01, 0.15), p=3.425e-02, d=0.21

## Ablation (if static_ga_experiment_log.csv present)
- Static GA lines cleared: 31.25 (vs Adaptive 79.06, Baseline 56.48)
### Static vs Adaptive (Welch)
* total_lines_cleared: diff=-47.81 (95% CI -58.04,-37.57), p=6.765e-18, d=-0.93
* max_score: diff=-4780.50 (95% CI -5803.63,-3757.37), p=6.765e-18, d=-0.93
* game_duration: diff=0.04 (95% CI 0.03,0.05), p=7.973e-12, d=1.24
* decision_latency: diff=4.66 (95% CI 4.53,4.78), p=1.971e-123, d=9.88
### Baseline vs Static (Welch)
* total_lines_cleared: diff=-25.23 (95% CI -36.91,-13.55), p=3.344e-05, d=-0.60
* max_score: diff=-2523.00 (95% CI -3690.74,-1355.26), p=3.344e-05, d=-0.60
* game_duration: diff=0.05 (95% CI 0.04,0.06), p=1.389e-14, d=1.25
* decision_latency: diff=4.73 (95% CI 4.62,4.85), p=6.981e-108, d=11.29