# Training Feedback

## Final-policy outcome
score=127.670240, len=316.900000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[0.773557, 310.128093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 84.000000 | 93.2% | 93.2% | 1.3% |
| crate_progress | 3.631770 | 4.0% | 4.3% | 71.7% |
| cart_approach | 1.119975 | 1.2% | 2.1% | 99.5% |
| shove | -0.253255 | -0.3% | 0.3% | 2.9% |
| gentleness_obs | -0.129781 | -0.1% | 0.1% | 39.2% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
