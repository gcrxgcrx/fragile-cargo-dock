# Training Feedback

## Final-policy outcome
score=2.988338, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.324884, 3.703213]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_settling | 348.727567 | 36.2% | 36.2% | 70.0% |
| joint_completion_proxy | 312.770581 | 32.4% | 32.4% | 68.2% |
| crate_dock_alignment | 247.959511 | 25.7% | 25.7% | 70.0% |
| crate_to_dock_progress | 53.467329 | 5.5% | 5.6% | 39.3% |
| gentle_contact | -1.348769 | -0.1% | 0.1% | 1.2% |
| wall_proximity | -0.000277 | -0.0% | 0.0% | 0.0% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
