# Training Feedback

## Final-policy outcome
score=-10.243742, len=393.300000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-103.266653, 0.838033]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_docking_quality | 992.014857 | 98.7% | 98.7% | 100.0% |
| crate_to_dock_progress | 0.207048 | 0.0% | 0.9% | 31.7% |
| action_smoothness | -2.597975 | -0.3% | 0.3% | 100.0% |
| out_of_bounds_penalty | -1.365637 | -0.1% | 0.1% | 0.5% |
| soft_contact_penalty | -0.153109 | -0.0% | 0.0% | 2.0% |
| crate_speed_penalty_near_dock | -0.073651 | -0.0% | 0.0% | 2.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
