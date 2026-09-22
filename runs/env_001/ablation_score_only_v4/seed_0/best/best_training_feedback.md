# Training Feedback

## Final-policy outcome
score=208.814713, len=573.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[176.878242, 240.771686]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_gate | 331.976252 | 95.2% | 95.2% | 98.5% |
| landing_reward | 14.837461 | 4.3% | 4.3% | 5.9% |
| progress_reward | 1.349368 | 0.4% | 0.4% | 97.4% |
| velocity_penalty | -0.264988 | -0.1% | 0.1% | 3.7% |
| angular_penalty | -0.179499 | -0.1% | 0.1% | 97.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
