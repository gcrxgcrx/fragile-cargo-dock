# Training Feedback

## Final-policy outcome
score=-122.786648, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.466519, -100.568487]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| orientation_penalty | -1.238043 | -47.4% | 47.4% | 100.0% |
| progress | 1.037906 | 39.7% | 42.7% | 100.0% |
| fuel_cost | -0.145000 | -5.5% | 5.5% | 3.9% |
| crash_prevention | -0.108674 | -4.2% | 4.2% | 6.1% |
| landing_reward | 0.006066 | 0.2% | 0.2% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
