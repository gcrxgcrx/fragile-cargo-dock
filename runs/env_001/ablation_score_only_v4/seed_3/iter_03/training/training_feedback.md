# Training Feedback

## Final-policy outcome
score=-111.708968, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.655212, -95.059093]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_penalty | -12.520726 | -79.9% | 79.9% | 100.0% |
| proximity_reward | 1.855074 | 11.8% | 12.3% | 100.0% |
| safe_contact_bonus | 1.214658 | 7.8% | 7.8% | 1.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
