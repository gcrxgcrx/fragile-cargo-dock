# Training Feedback

## Final-policy outcome
score=-119.472383, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.892654, -95.474620]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -64.065840 | -82.2% | 82.2% | 100.0% |
| soft_landing_penalty | -12.857088 | -16.5% | 16.5% | 100.0% |
| safe_contact_bonus | 0.972959 | 1.2% | 1.2% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
