# Training Feedback

## Final-policy outcome
score=23.437457, len=396.550000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[1.309697, 309.832763]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 31.000000 | 82.4% | 82.4% | 0.4% |
| crate_progress | 3.478512 | 9.3% | 11.7% | 85.1% |
| cart_approach | 1.266480 | 3.4% | 5.3% | 97.9% |
| roughness | -0.206083 | -0.5% | 0.5% | 34.8% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
