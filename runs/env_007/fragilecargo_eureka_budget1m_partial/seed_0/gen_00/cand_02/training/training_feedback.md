# Training Feedback

## Final-policy outcome
score=3.574028, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.090709, 8.584942]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settle_proxy | 252.964954 | 92.1% | 92.1% | 56.3% |
| crate_dock_progress | 20.370730 | 7.4% | 7.8% | 46.5% |
| fragile_impact_guard | -0.249763 | -0.1% | 0.1% | 0.2% |
| cart_bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |
| crate_bounds_guard | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
