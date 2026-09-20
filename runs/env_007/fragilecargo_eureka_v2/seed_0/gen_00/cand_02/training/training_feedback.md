# Training Feedback

## Final-policy outcome
score=3.679573, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.566374, 8.049696]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_proxy_reward | 318.500104 | 91.8% | 91.8% | 70.7% |
| gentle_contact_penalty | -11.835669 | -3.4% | 3.4% | 15.7% |
| crate_to_dock_progress | 10.987357 | 3.2% | 3.3% | 43.4% |
| action_smoothness_penalty | -3.810107 | -1.1% | 1.1% | 100.0% |
| boundary_penalty | -1.241645 | -0.4% | 0.4% | 9.7% |
| crate_dock_alignment | -0.011430 | -0.0% | 0.0% | 41.6% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
