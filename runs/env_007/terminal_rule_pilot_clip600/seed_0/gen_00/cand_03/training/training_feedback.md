# Training Feedback

## Final-policy outcome
score=2.885780, len=397.300000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-101.479245, 9.602852]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| dock_event | 7500.000000 | 96.2% | 96.2% | 3.1% |
| crate_align | 197.397094 | 2.5% | 2.5% | 36.4% |
| crate_progress | 46.556303 | 0.6% | 0.6% | 85.4% |
| soft_contact | -40.989427 | -0.5% | 0.5% | 49.4% |
| crate_speed_pen | -11.259276 | -0.1% | 0.1% | 12.5% |
| action_smooth | -2.307908 | -0.0% | 0.0% | 100.0% |
| out_of_bounds | -0.235649 | -0.0% | 0.0% | 0.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
