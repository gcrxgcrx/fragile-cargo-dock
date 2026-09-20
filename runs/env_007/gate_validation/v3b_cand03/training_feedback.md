# Training Feedback

## Final-policy outcome
score=2.779201, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.582762, 3.581719]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_progress | 11.924955 | 79.5% | 80.2% | 40.5% |
| speed_near_dock | -1.237814 | -8.3% | 8.3% | 1.0% |
| action_smoothness | -1.000924 | -6.7% | 6.7% | 100.0% |
| align_near_dock | -0.520920 | -3.5% | 3.5% | 2.7% |
| gentleness | -0.211763 | -1.4% | 1.4% | 10.3% |
| enter_dock_event | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
