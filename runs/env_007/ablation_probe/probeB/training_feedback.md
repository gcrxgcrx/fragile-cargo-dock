# Training Feedback

## Final-policy outcome
score=68.039583, len=361.650000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-0.521396, 310.075293]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 60.000000 | 86.1% | 86.1% | 0.1% |
| dock_enter | 4.250000 | 6.1% | 6.1% | 0.2% |
| crate_progress | 3.620574 | 5.2% | 5.5% | 56.2% |
| cart_approach | 1.221926 | 1.8% | 2.2% | 95.3% |
| shove | -0.092192 | -0.1% | 0.1% | 1.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
