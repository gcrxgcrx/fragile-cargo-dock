# Training Feedback

## Final-policy outcome
score=0.337006, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.099239, 1.196229]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| REPAIR_cart_approach | 3.287806 | 63.8% | 88.4% | 100.0% |
| dock_speed_penalty | -0.327145 | -6.4% | 6.4% | 2.9% |
| REPAIR_crate_progress | 0.191226 | 3.7% | 3.7% | 11.7% |
| soft_contact | -0.039239 | -0.8% | 0.8% | 0.2% |
| crate_to_dock_progress | 0.038813 | 0.8% | 0.8% | 11.7% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
