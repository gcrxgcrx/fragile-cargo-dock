# Training Feedback

## Final-policy outcome
score=264.252779, len=207.450000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[8.464494, 309.924833]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 179.000000 | 96.1% | 96.1% | 4.3% |
| crate_progress | 3.969711 | 2.1% | 2.3% | 68.7% |
| cart_approach | 0.924358 | 0.5% | 1.0% | 98.3% |
| shove | -1.250991 | -0.7% | 0.7% | 19.5% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
