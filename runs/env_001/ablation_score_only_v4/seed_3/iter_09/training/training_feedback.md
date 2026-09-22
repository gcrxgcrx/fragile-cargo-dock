# Training Feedback

## Final-policy outcome
score=-12.711452, len=964.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-53.553015, 249.186669]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_approach | 1628.787230 | 49.6% | 49.6% | 100.0% |
| distance_shaping | 1598.020612 | 48.7% | 48.7% | 100.0% |
| efficiency_penalty | -39.787290 | -1.2% | 1.2% | 99.4% |
| progress_reward | 11.403805 | 0.3% | 0.5% | 99.9% |
| orientation_penalty | -0.949711 | -0.0% | 0.0% | 99.7% |
| velocity_constraint | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
