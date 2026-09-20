# EUREKA-style population search - summary

- environment: `FragileCargoDock-v0`
- generations: 1, population 4 (elite 2 x 1 + 1 children each)
- timesteps per candidate: 1,200,000
- evaluation episodes: 20
- selection metric: native `external_eval.mean_eval_reward`
- seed: 0

**Best candidate: `g00c03` with native score 34.8838**

## Generation 0

| id | parent | native score |
|---|---|---:|
| g00c03 | - | 34.8838 |
| g00c01 | - | -1.7296 |
| g00c00 | - | -20.5170 |
| g00c02 | - | -102.9976 |

## Best reward

`runs\env_007\terminal_rule_pilot\seed_0\gen_00\cand_03\reward_v1.py`

copied to `runs\env_007\terminal_rule_pilot\seed_0\best\best_reward.py`