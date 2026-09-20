# Training Feedback

## Final-policy outcome
score=3.047868, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-0.515163, 8.066750]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 2746.715703 | 98.1% | 98.1% | 44.6% |
| speed_penalty_near_dock | -43.469891 | -1.6% | 1.6% | 4.7% |
| soft_contact_gentleness | -5.891095 | -0.2% | 0.2% | 24.6% |
| first_entry_bonus | 4.500000 | 0.2% | 0.2% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| settled_hold_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
