# Training Feedback

## Final-policy outcome
score=11.177912, len=388.150000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-104.628499, 309.149267]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 8.475301 | 72.9% | 75.3% | 41.0% |
| completion_improvement | 2.366467 | 20.4% | 21.7% | 32.2% |
| soft_contact_penalty | -0.325057 | -2.8% | 2.8% | 6.0% |
| boundary_avoidance | -0.026897 | -0.2% | 0.2% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
