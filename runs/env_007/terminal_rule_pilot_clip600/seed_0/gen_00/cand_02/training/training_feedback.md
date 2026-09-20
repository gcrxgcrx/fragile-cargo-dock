# Training Feedback

## Final-policy outcome
score=-102.997620, len=124.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-104.174157, -101.004697]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| action_smoothness | -0.951560 | -55.4% | 55.4% | 100.0% |
| oob_penalty | -0.572339 | -33.3% | 33.3% | 9.0% |
| time_penalty | -0.194005 | -11.3% | 11.3% | 99.2% |
| completion_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_dock_quality | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
