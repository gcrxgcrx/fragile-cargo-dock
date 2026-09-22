# Training Feedback

## Final-policy outcome
score=-121.771090, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.398763, -100.643724]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.055179 | 46.2% | 49.7% | 100.0% |
| orientation_penalty | -1.034959 | -45.3% | 45.3% | 100.0% |
| fuel_cost | -0.115000 | -5.0% | 5.0% | 3.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
