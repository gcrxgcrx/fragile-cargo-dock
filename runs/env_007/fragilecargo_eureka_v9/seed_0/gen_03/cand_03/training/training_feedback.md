# Training Feedback

## Final-policy outcome
score=278.713612, len=197.500000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-0.331487, 310.359257]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 270.000000 | 87.3% | 87.3% | 0.5% |
| settle_bonus | 20.948002 | 6.8% | 6.8% | 16.5% |
| progress | 15.696856 | 5.1% | 5.1% | 71.3% |
| approach_cargo | 0.933412 | 0.3% | 0.7% | 96.5% |
| time_cost | -0.395000 | -0.1% | 0.1% | 100.0% |
| action_cost | -0.087420 | -0.0% | 0.0% | 100.0% |
| bounds_safety | 0.000000 | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| roughness | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
