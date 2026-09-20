# Training Feedback

## Final-policy outcome
score=218.957022, len=253.950000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[3.762205, 309.947629]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 210.000000 | 95.3% | 95.3% | 0.3% |
| dock_enter | 4.500000 | 2.0% | 2.0% | 0.4% |
| crate_progress | 4.030819 | 1.8% | 1.9% | 65.2% |
| cart_approach | 1.146856 | 0.5% | 0.7% | 99.4% |
| shove | -0.069552 | -0.0% | 0.0% | 2.8% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
