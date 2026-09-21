# Training Feedback

## Final-policy outcome
score=278.358414, len=221.300000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[0.221401, 310.149569]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 272.000000 | 97.2% | 97.2% | 6.1% |
| crate_progress | 3.884447 | 1.4% | 1.4% | 74.5% |
| cart_approach | 0.756976 | 0.3% | 1.1% | 97.0% |
| shove | -1.040581 | -0.4% | 0.4% | 15.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
