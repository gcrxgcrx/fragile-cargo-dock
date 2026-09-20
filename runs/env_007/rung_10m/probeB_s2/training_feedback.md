# Training Feedback

## Final-policy outcome
score=234.138680, len=240.000000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[4.076016, 310.034172]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 225.000000 | 95.2% | 95.2% | 0.3% |
| dock_enter | 4.750000 | 2.0% | 2.0% | 0.4% |
| crate_progress | 3.995350 | 1.7% | 1.7% | 64.7% |
| cart_approach | 1.086943 | 0.5% | 0.7% | 98.8% |
| shove | -0.979612 | -0.4% | 0.4% | 15.8% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
