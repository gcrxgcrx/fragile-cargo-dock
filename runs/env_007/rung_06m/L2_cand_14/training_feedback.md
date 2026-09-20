# Training Feedback

## Final-policy outcome
score=-16.403702, len=366.450000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-102.630685, -0.071065]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| bounds_guard | -16.561711 | -95.1% | 95.1% | 2.8% |
| crate_dock_progress | 0.822155 | 4.7% | 4.7% | 2.6% |
| gentleness | -0.022876 | -0.1% | 0.1% | 0.1% |
| dock_quality_gate | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
