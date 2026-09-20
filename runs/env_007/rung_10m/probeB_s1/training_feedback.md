# Training Feedback

## Final-policy outcome
score=309.647410, len=198.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.234736, 310.488603]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 96.6% | 96.6% | 0.5% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.5% |
| crate_progress | 4.034151 | 1.3% | 1.3% | 79.3% |
| cart_approach | 1.241856 | 0.4% | 0.5% | 99.4% |
| shove | -0.048047 | -0.0% | 0.0% | 3.1% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
