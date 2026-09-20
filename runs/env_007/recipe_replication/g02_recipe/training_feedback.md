# Training Feedback

## Final-policy outcome
score=173.923761, len=315.050000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[0.849034, 310.079370]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_progress | 242.467468 | 42.6% | 42.6% | 53.5% |
| success_event | 165.000000 | 29.0% | 29.0% | 0.2% |
| REPAIR_settled_stream | 133.000000 | 23.3% | 23.3% | 2.1% |
| enter_event | 28.500000 | 5.0% | 5.0% | 0.3% |
| gentleness | -0.699008 | -0.1% | 0.1% | 18.2% |
| dock_quality_gate | 0.000084 | 0.0% | 0.0% | 0.8% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
