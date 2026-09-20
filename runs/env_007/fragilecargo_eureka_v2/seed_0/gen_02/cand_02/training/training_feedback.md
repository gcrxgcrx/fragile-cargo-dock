# Training Feedback

## Final-policy outcome
score=3.748833, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.092929, 5.010998]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_proxy_reward | 88.569299 | 59.8% | 59.8% | 59.6% |
| crate_to_dock_progress | 49.433045 | 33.4% | 33.5% | 85.0% |
| gentle_contact_penalty | -6.025915 | -4.1% | 4.1% | 13.5% |
| action_smoothness_penalty | -2.623706 | -1.8% | 1.8% | 100.0% |
| boundary_penalty | -0.955946 | -0.6% | 0.6% | 12.6% |
| crate_dock_alignment | -0.245860 | -0.2% | 0.2% | 98.0% |
| obstacle_penalty | -0.023190 | -0.0% | 0.0% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
