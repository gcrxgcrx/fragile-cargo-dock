# Training Feedback

## Final-policy outcome
score=-11.882732, len=386.550000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-102.755584, 0.982687]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| bounds_guard | -2.494830 | -69.6% | 69.6% | 0.5% |
| crate_to_dock_progress | 0.883456 | 24.7% | 24.7% | 4.0% |
| dock_speed_penalty | -0.205690 | -5.7% | 5.7% | 1.8% |
| OVERSHOOT_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_contact | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
