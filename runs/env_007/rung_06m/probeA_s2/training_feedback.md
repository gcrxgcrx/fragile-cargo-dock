# Training Feedback

## Final-policy outcome
score=29.022086, len=376.550000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-97.962213, 309.141931]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 20.000000 | 73.3% | 73.3% | 0.3% |
| crate_progress | 3.066649 | 11.2% | 16.9% | 74.7% |
| cart_approach | 0.909347 | 3.3% | 8.2% | 99.5% |
| roughness | -0.306317 | -1.1% | 1.1% | 43.5% |
| shove | -0.116755 | -0.4% | 0.4% | 1.6% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
