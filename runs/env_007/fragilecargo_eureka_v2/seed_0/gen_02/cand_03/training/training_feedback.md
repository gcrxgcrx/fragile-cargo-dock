# Training Feedback

## Final-policy outcome
score=8.092716, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[2.064939, 9.922352]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_proxy_reward | 1194.937989 | 89.5% | 89.5% | 65.9% |
| crate_dock_alignment | 108.522697 | 8.1% | 8.1% | 74.9% |
| crate_to_dock_progress | 15.795772 | 1.2% | 1.2% | 44.6% |
| gentle_contact_penalty | -11.608480 | -0.9% | 0.9% | 14.7% |
| action_smoothness_penalty | -3.338379 | -0.3% | 0.3% | 100.0% |
| boundary_penalty | -0.895178 | -0.1% | 0.1% | 11.8% |
| obstacle_penalty | -0.060379 | -0.0% | 0.0% | 1.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
