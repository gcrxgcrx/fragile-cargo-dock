# Training Feedback

## Final-policy outcome
score=2.481639, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.458022, 6.961918]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| docked_settle | 402.977248 | 84.6% | 84.6% | 42.3% |
| crate_progress_toward_dock | 53.127205 | 11.2% | 11.6% | 39.9% |
| settle_speed_penalty | -10.743974 | -2.3% | 2.3% | 23.8% |
| cart_crate_approach | -0.420183 | -0.1% | 1.3% | 68.9% |
| fragile_handling_penalty | -1.195730 | -0.3% | 0.3% | 3.2% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
