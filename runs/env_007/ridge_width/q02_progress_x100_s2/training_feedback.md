# Training Feedback

## Final-policy outcome
score=9.219725, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[8.694137, 10.336977]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| REPAIR_settled_stream | 502.000000 | 46.5% | 46.5% | 6.3% |
| crate_to_dock_progress | 447.300009 | 41.4% | 41.4% | 55.7% |
| dock_speed_penalty | -99.604746 | -9.2% | 9.2% | 71.2% |
| enter_dock_event | 30.000000 | 2.8% | 2.8% | 0.2% |
| soft_contact | -1.439166 | -0.1% | 0.1% | 15.4% |
| bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
