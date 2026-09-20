# Training Feedback

## Final-policy outcome
score=173.081081, len=328.700000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[0.437458, 309.859848]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 180.000000 | 96.6% | 96.6% | 2.7% |
| crate_progress | 3.782914 | 2.0% | 2.1% | 83.6% |
| cart_approach | 1.244147 | 0.7% | 1.0% | 99.8% |
| shove | -0.393958 | -0.2% | 0.2% | 5.3% |
| roughness | -0.202185 | -0.1% | 0.1% | 40.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
