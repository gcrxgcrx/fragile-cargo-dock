# Training Feedback

## Final-policy outcome
score=53.015391, len=369.100000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[0.387102, 309.421912]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_progress | 21.092807 | 40.9% | 54.6% | 45.0% |
| enter_event | 13.500000 | 26.2% | 26.2% | 0.2% |
| success_event | 9.000000 | 17.4% | 17.4% | 0.0% |
| gentleness | -0.873953 | -1.7% | 1.7% | 4.7% |
| dock_quality_gate | -0.068629 | -0.1% | 0.1% | 9.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
