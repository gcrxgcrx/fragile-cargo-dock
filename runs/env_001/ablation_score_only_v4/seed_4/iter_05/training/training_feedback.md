# Training Feedback

## Final-policy outcome
score=-115.805737, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.692627, -96.181510]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| potential_gain | 0.845976 | 54.2% | 54.5% | 3.1% |
| safe_landing_penalty | -0.678016 | -43.4% | 43.4% | 100.0% |
| action_efficiency_cost | -0.031500 | -2.0% | 2.0% | 4.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
