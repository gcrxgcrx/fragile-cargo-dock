# Training Feedback

## Final-policy outcome
score=1.296805, len=395.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-98.345217, 9.874608]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 3.149621 | 49.3% | 65.3% | 85.9% |
| cart_approach | 1.241874 | 19.4% | 26.4% | 98.0% |
| gentleness_obs | -0.369107 | -5.8% | 5.8% | 32.3% |
| shove | -0.158523 | -2.5% | 2.5% | 2.3% |
| boundary | -0.000011 | -0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
