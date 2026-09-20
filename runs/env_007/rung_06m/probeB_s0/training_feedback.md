# Training Feedback

## Final-policy outcome
score=81.689552, len=356.300000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-0.022783, 309.688629]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 75.000000 | 88.4% | 88.4% | 0.1% |
| crate_progress | 3.422242 | 4.0% | 5.0% | 76.6% |
| dock_enter | 3.250000 | 3.8% | 3.8% | 0.2% |
| cart_approach | 1.124103 | 1.3% | 2.6% | 99.5% |
| shove | -0.106141 | -0.1% | 0.1% | 2.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
