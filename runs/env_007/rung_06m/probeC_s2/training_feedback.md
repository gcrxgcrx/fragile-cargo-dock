# Training Feedback

## Final-policy outcome
score=8.147983, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[4.067026, 10.079856]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 3.455475 | 31.7% | 42.4% | 61.0% |
| dock_enter | 4.500000 | 41.3% | 41.3% | 0.2% |
| cart_approach | 1.198324 | 11.0% | 14.9% | 97.4% |
| roughness | -0.155660 | -1.4% | 1.4% | 39.9% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
