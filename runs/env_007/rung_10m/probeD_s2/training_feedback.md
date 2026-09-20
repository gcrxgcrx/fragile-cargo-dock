# Training Feedback

## Final-policy outcome
score=7.365257, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.126872, 9.525680]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 28.000000 | 81.7% | 81.7% | 0.4% |
| crate_progress | 3.693065 | 10.8% | 11.7% | 83.2% |
| cart_approach | 1.207811 | 3.5% | 5.5% | 99.2% |
| gentleness_obs | -0.184556 | -0.5% | 0.5% | 23.7% |
| shove | -0.183211 | -0.5% | 0.5% | 2.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
