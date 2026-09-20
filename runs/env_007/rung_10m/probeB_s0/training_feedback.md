# Training Feedback

## Final-policy outcome
score=248.580833, len=264.650000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-0.302396, 310.557537]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 240.000000 | 95.8% | 95.8% | 0.3% |
| dock_enter | 4.500000 | 1.8% | 1.8% | 0.3% |
| crate_progress | 3.767570 | 1.5% | 1.6% | 76.9% |
| cart_approach | 1.172588 | 0.5% | 0.7% | 97.7% |
| shove | -0.305576 | -0.1% | 0.1% | 7.0% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_settled_hold | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
