# Training Feedback

## Final-policy outcome
score=5.665167, len=84.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-50.492260, 39.885759]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -71.683724 | -81.1% | 81.1% | 100.0% |
| soft_landing_penalty | -14.206739 | -16.1% | 16.1% | 56.9% |
| safe_contact_bonus | 2.479459 | 2.8% | 2.8% | 3.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
