# Training Feedback

## Final-policy outcome
score=-118.519732, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.814028, -96.181510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.062790 | 43.4% | 46.8% | 100.0% |
| orientation_penalty | -1.075724 | -44.0% | 44.0% | 100.0% |
| fuel_cost | -0.115000 | -4.7% | 4.7% | 3.4% |
| crash_prevention | -0.111566 | -4.6% | 4.6% | 6.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
