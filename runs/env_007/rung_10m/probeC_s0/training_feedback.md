# Training Feedback

## Final-policy outcome
score=294.341804, len=249.550000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[4.019288, 310.671694]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 285.000000 | 96.3% | 96.3% | 0.4% |
| dock_enter | 4.750000 | 1.6% | 1.6% | 0.4% |
| crate_progress | 4.047471 | 1.4% | 1.4% | 84.2% |
| cart_approach | 1.310083 | 0.4% | 0.6% | 99.9% |
| roughness | -0.200989 | -0.1% | 0.1% | 43.1% |
| shove | -0.148676 | -0.1% | 0.1% | 4.0% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
