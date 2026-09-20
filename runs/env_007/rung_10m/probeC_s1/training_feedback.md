# Training Feedback

## Final-policy outcome
score=294.392714, len=208.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[4.951537, 309.987340]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 285.000000 | 96.4% | 96.4% | 0.5% |
| dock_enter | 4.750000 | 1.6% | 1.6% | 0.5% |
| crate_progress | 4.043684 | 1.4% | 1.4% | 79.2% |
| cart_approach | 1.235418 | 0.4% | 0.5% | 99.3% |
| roughness | -0.161320 | -0.1% | 0.1% | 50.2% |
| shove | -0.100215 | -0.0% | 0.0% | 3.5% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
