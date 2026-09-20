# Training Feedback

## Final-policy outcome
score=36.981953, len=382.250000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[0.883904, 309.939095]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 31.000000 | 84.5% | 84.5% | 0.4% |
| crate_progress | 2.943640 | 8.0% | 10.5% | 70.9% |
| cart_approach | 1.239766 | 3.4% | 4.6% | 95.6% |
| roughness | -0.142891 | -0.4% | 0.4% | 60.4% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
