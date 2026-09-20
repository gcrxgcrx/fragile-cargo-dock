# Training Feedback

## Final-policy outcome
score=4.932269, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.482116, 8.711004]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 774.199058 | 85.0% | 85.0% | 24.8% |
| dock_speed_penalty | -124.218755 | -13.6% | 13.6% | 32.5% |
| enter_dock_event | 10.500000 | 1.2% | 1.2% | 0.1% |
| soft_contact | -1.933968 | -0.2% | 0.2% | 11.4% |
| REPAIR_gentleness | -0.079934 | -0.0% | 0.0% | 11.4% |
| REPAIR_settled_stream | 0.000000 | 0.0% | 0.0% | 0.0% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
