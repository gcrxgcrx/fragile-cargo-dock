# Training Feedback

## Final-policy outcome
score=1.698421, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.877702, 2.868367]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 2.135394 | 72.1% | 72.1% | 48.6% |
| align_gated_progress | 0.687171 | 23.2% | 23.2% | 48.6% |
| speed_near_dock_penalty | -0.096790 | -3.3% | 3.3% | 6.9% |
| gentleness | -0.041893 | -1.4% | 1.4% | 8.1% |
| enter_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
