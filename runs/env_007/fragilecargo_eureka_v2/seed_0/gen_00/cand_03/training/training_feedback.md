# Training Feedback

## Final-policy outcome
score=3.017389, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.891357, 6.687076]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_completion_proxy | 571.005911 | 54.8% | 54.8% | 75.8% |
| crate_settling | 246.273520 | 23.6% | 23.6% | 80.6% |
| crate_dock_alignment | 198.617685 | 19.1% | 19.1% | 80.6% |
| crate_to_dock_progress | 21.959838 | 2.1% | 2.3% | 41.8% |
| gentle_contact | -1.910072 | -0.2% | 0.2% | 2.1% |
| wall_proximity | -0.001145 | -0.0% | 0.0% | 0.2% |
| boundary_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
