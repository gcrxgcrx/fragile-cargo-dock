# Training Feedback

## Final-policy outcome
score=50.659456, len=703.500000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-91.862116, 175.714803]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_shaping | 1093.129434 | 62.4% | 62.4% | 100.0% |
| landing_bonus | 598.500000 | 34.2% | 34.2% | 2.8% |
| fuel_penalty | -41.690000 | -2.4% | 2.4% | 89.2% |
| progress_reward | 9.027308 | 0.5% | 0.7% | 100.0% |
| orientation_penalty | -5.563954 | -0.3% | 0.3% | 100.0% |
| velocity_constraint | -0.021983 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
