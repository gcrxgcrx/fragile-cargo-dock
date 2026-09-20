# Training Feedback

## Final-policy outcome
score=218.836236, len=255.700000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[1.508812, 310.445989]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 210.000000 | 94.9% | 94.9% | 0.3% |
| dock_enter | 4.750000 | 2.1% | 2.1% | 0.4% |
| crate_progress | 3.856697 | 1.7% | 1.8% | 66.2% |
| cart_approach | 0.961997 | 0.4% | 0.8% | 99.6% |
| shove | -0.605689 | -0.3% | 0.3% | 11.7% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
