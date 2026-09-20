# Training Feedback

## Final-policy outcome
score=3.861396, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.278738, 7.650139]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_settling_and_alignment | 54.794585 | 76.9% | 76.9% | 79.7% |
| crate_to_dock_progress | 14.538438 | 20.4% | 21.7% | 47.2% |
| fragile_impact_penalty | -0.964330 | -1.4% | 1.4% | 0.5% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
