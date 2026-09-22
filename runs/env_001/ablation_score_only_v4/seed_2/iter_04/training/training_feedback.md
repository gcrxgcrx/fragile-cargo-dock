# Training Feedback

## Final-policy outcome
score=-119.706688, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-173.501598, 10.994076]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.105799 | 46.1% | 49.4% | 100.0% |
| orientation_penalty | -0.993071 | -41.4% | 41.4% | 100.0% |
| crash_prevention | -0.110848 | -4.6% | 4.6% | 6.1% |
| fuel_cost | -0.105000 | -4.4% | 4.4% | 3.1% |
| landing_reward | 0.006230 | 0.3% | 0.3% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
