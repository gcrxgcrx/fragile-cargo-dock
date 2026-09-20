# Training Feedback

## Final-policy outcome
score=-7.055429, len=400.000000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-104.137851, -0.332112]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 67.250968 | 49.3% | 49.3% | 100.0% |
| crate_settling | 66.916431 | 49.1% | 49.1% | 100.0% |
| boundary_avoidance | -1.892003 | -1.4% | 1.4% | 23.3% |
| action_smoothness | -0.268817 | -0.2% | 0.2% | 100.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| gentle_contact | 0.000000 | 0.0% | 0.0% | 0.0% |
| obstacle_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
