# Training Feedback

## Final-policy outcome
score=11.006412, len=780.500000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-79.395903, 153.616317]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 446.000794 | 78.9% | 78.9% | 100.0% |
| safe_contact_proxy | 105.202515 | 18.6% | 18.6% | 13.9% |
| velocity_damping | -11.095673 | -2.0% | 2.0% | 100.0% |
| angle_penalty | -2.797662 | -0.5% | 0.5% | 100.0% |
| angvel_penalty | -0.364033 | -0.1% | 0.1% | 94.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
