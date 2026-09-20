# Training Feedback

## Final-policy outcome
score=34.883847, len=394.950000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[1.055578, 309.642692]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_event | 750.000000 | 72.3% | 72.3% | 0.3% |
| crate_align | 173.129587 | 16.7% | 16.7% | 37.5% |
| soft_contact | -51.056738 | -4.9% | 4.9% | 48.6% |
| crate_progress | 42.855986 | 4.1% | 4.4% | 86.8% |
| crate_speed_pen | -14.805908 | -1.4% | 1.4% | 19.0% |
| action_smooth | -2.204229 | -0.2% | 0.2% | 100.0% |
| out_of_bounds | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
