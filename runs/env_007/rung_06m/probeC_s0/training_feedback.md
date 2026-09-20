# Training Feedback

## Final-policy outcome
score=202.623785, len=297.600000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[2.772065, 310.775487]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 195.000000 | 95.5% | 95.5% | 0.2% |
| crate_progress | 3.958829 | 1.9% | 1.9% | 75.1% |
| dock_enter | 3.250000 | 1.6% | 1.6% | 0.2% |
| cart_approach | 1.258314 | 0.6% | 0.9% | 99.4% |
| roughness | -0.191031 | -0.1% | 0.1% | 39.4% |
| shove | -0.033232 | -0.0% | 0.0% | 1.3% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
