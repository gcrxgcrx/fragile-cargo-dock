# Training Feedback

## Final-policy outcome
score=1.693049, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[0.156392, 3.935311]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_cargo | 1.154640 | 22.1% | 49.9% | 100.0% |
| progress | 1.550356 | 29.6% | 29.7% | 56.7% |
| time_cost | -0.800000 | -15.3% | 15.3% | 100.0% |
| roughness | -0.207716 | -4.0% | 4.0% | 4.7% |
| action_cost | -0.058277 | -1.1% | 1.1% | 100.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| dock_enter | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| settle_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
