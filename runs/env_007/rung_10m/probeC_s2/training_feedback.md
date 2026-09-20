# Training Feedback

## Final-policy outcome
score=263.749014, len=289.700000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[3.863362, 309.919538]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 255.000000 | 96.2% | 96.2% | 0.3% |
| dock_enter | 4.250000 | 1.6% | 1.6% | 0.3% |
| crate_progress | 3.978925 | 1.5% | 1.5% | 81.9% |
| cart_approach | 1.303725 | 0.5% | 0.6% | 99.2% |
| roughness | -0.151978 | -0.1% | 0.1% | 48.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
