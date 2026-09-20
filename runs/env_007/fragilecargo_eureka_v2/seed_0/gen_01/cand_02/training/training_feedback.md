# Training Feedback

## Final-policy outcome
score=3.705112, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.376224, 4.237008]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_proxy_reward | 126.384235 | 76.4% | 76.4% | 62.5% |
| crate_to_dock_progress | 27.838678 | 16.8% | 17.6% | 87.3% |
| gentle_contact_penalty | -6.413006 | -3.9% | 3.9% | 13.7% |
| action_smoothness_penalty | -2.805544 | -1.7% | 1.7% | 100.0% |
| boundary_penalty | -0.633914 | -0.4% | 0.4% | 8.3% |
| crate_dock_alignment | -0.036026 | -0.0% | 0.1% | 86.7% |
| obstacle_penalty | -0.000581 | -0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
