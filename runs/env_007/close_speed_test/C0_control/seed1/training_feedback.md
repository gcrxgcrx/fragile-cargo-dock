# Training Feedback

## Final-policy outcome
score=263.842421, len=207.700000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[2.708352, 310.598209]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 170.000000 | 95.6% | 95.6% | 4.1% |
| crate_progress | 4.012422 | 2.3% | 2.3% | 66.6% |
| cart_approach | 0.998040 | 0.6% | 1.5% | 99.5% |
| shove | -1.193793 | -0.7% | 0.7% | 18.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
