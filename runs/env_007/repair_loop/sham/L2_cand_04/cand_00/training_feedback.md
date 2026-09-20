# Training Feedback

## Final-policy outcome
score=3.176396, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-1.491629, 9.043988]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 3714.673589 | 98.4% | 98.4% | 100.0% |
| entry_bonus | 30.000000 | 0.8% | 0.8% | 0.4% |
| dock_quality | 29.092327 | 0.8% | 0.8% | 1.2% |
| action_penalty | -2.698210 | -0.1% | 0.1% | 100.0% |
| speed_gate_penalty | -0.472670 | -0.0% | 0.0% | 0.4% |
| bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_event | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
