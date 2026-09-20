# Training Feedback

## Final-policy outcome
score=2.449713, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.536395, 7.496496]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_proxy_reward | 1206.726588 | 85.6% | 85.6% | 59.3% |
| crate_dock_alignment | 145.330519 | 10.3% | 10.3% | 73.1% |
| crate_to_dock_progress | 28.936246 | 2.1% | 2.1% | 41.5% |
| gentle_contact_penalty | -12.486778 | -0.9% | 0.9% | 16.6% |
| time_penalty | -9.975000 | -0.7% | 0.7% | 99.8% |
| action_smoothness_penalty | -3.396016 | -0.2% | 0.2% | 100.0% |
| obstacle_penalty | -1.957599 | -0.1% | 0.1% | 13.0% |
| boundary_penalty | -0.638742 | -0.0% | 0.0% | 8.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
