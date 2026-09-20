# Training Feedback

## Final-policy outcome
score=1.454337, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.066271, 2.581595]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_proxy_reward | 379.429491 | 90.2% | 90.2% | 44.7% |
| crate_to_dock_progress | 18.437399 | 4.4% | 5.0% | 42.7% |
| gentle_contact_penalty | -9.806503 | -2.3% | 2.3% | 14.8% |
| crate_dock_alignment | -6.628385 | -1.6% | 1.6% | 13.6% |
| boundary_penalty | -1.509642 | -0.4% | 0.4% | 11.5% |
| obstacle_penalty | -1.189656 | -0.3% | 0.3% | 7.8% |
| action_smoothness_penalty | -1.092768 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
