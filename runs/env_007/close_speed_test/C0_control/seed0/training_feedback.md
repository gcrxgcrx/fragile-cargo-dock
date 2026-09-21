# Training Feedback

## Final-policy outcome
score=8.382554, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.611837, 9.874629]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 68.000000 | 89.8% | 89.8% | 0.9% |
| crate_progress | 3.537576 | 4.7% | 5.7% | 87.4% |
| cart_approach | 1.260705 | 1.7% | 3.2% | 99.4% |
| shove | -1.038680 | -1.4% | 1.4% | 9.2% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
