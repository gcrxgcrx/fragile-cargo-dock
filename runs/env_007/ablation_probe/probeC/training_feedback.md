# Training Feedback

## Final-policy outcome
score=309.588542, len=206.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.192852, 310.706882]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 96.4% | 96.4% | 0.5% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.5% |
| crate_progress | 4.036370 | 1.3% | 1.3% | 83.0% |
| cart_approach | 1.232526 | 0.4% | 0.6% | 100.0% |
| roughness | -0.204262 | -0.1% | 0.1% | 49.2% |
| shove | -0.081113 | -0.0% | 0.0% | 4.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
