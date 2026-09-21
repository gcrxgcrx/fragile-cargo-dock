# Training Feedback

## Final-policy outcome
score=112.822299, len=396.000000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[0.310268, 309.356935]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 380.000000 | 98.5% | 98.5% | 4.8% |
| crate_progress | 3.817216 | 1.0% | 1.0% | 87.3% |
| cart_approach | 1.298802 | 0.3% | 0.5% | 98.4% |
| gentleness_obs | -0.245457 | -0.1% | 0.1% | 26.8% |
| shove | -0.008627 | -0.0% | 0.0% | 0.3% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
