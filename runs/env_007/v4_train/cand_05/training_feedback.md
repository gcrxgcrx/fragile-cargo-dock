# Training Feedback

## Final-policy outcome
score=7.330711, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[1.604883, 8.621522]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_to_dock_progress | 31.308742 | 85.7% | 85.7% | 31.9% |
| enter_dock_once | 4.750000 | 13.0% | 13.0% | 0.2% |
| crate_speed_near_dock | -0.271674 | -0.7% | 0.7% | 1.1% |
| soft_contact | -0.120788 | -0.3% | 0.3% | 23.1% |
| out_of_bounds | -0.065765 | -0.2% | 0.2% | 0.9% |
| obstacle | -0.000423 | -0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
