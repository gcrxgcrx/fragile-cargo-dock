# Training Feedback

## Final-policy outcome
score=4.293235, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.924413, 8.599042]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| APPROACH_funnel | 1248.200004 | 80.9% | 80.9% | 73.5% |
| crate_to_dock_progress | 190.984999 | 12.4% | 12.4% | 38.0% |
| dock_speed_penalty | -98.124981 | -6.4% | 6.4% | 33.3% |
| enter_dock_event | 3.000000 | 0.2% | 0.2% | 0.0% |
| soft_contact | -2.081575 | -0.1% | 0.1% | 13.0% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
