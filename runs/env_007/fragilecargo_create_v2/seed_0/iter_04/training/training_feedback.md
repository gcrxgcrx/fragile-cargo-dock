# Training Feedback

## Final-policy outcome
score=17.693928, len=392.350000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-1.494248, 308.577205]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_gate | 233.336679 | 95.3% | 95.3% | 71.0% |
| crate_to_dock_progress | 11.354499 | 4.6% | 4.6% | 42.6% |
| soft_contact_penalty | -0.265219 | -0.1% | 0.1% | 4.3% |
| boundary_avoidance | -0.000274 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
