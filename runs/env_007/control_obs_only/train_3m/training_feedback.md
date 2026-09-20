# Training Feedback

## Final-policy outcome
score=22.297633, len=391.350000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-0.424125, 309.558340]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_settled_hold | 112.000000 | 93.7% | 93.7% | 1.4% |
| crate_progress | 2.684311 | 2.2% | 3.9% | 82.5% |
| cart_approach | 1.275421 | 1.1% | 2.4% | 97.1% |
| boundary | 0.000000 | 0.0% | 0.0% | 0.0% |
| shove | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
