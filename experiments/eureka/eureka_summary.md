# EUREKA-style population search - summary

- environment: `FragileCargoDock-v0`
- generations: 4, population 4 (elite 2 x 1 + 1 children each)
- timesteps per candidate: 3,000,000
- evaluation episodes: 20
- selection metric: native `external_eval.mean_eval_reward`
- seed: 0

**Best candidate: `g02c03` with native score 5.1258**

## Generation 0

| id | parent | native score |
|---|---|---:|
| g00c01 | - | -1.8634 |
| g00c03 | - | -2.4734 |
| g00c02 | - | -7.0081 |
| g00c00 | - | -102.2409 |

## Generation 1

| id | parent | native score |
|---|---|---:|
| g01c03 | g00c03 | 3.5500 |
| g01c02 | g00c01 | -1.3397 |
| g01e00 | g00c01 | -1.8634 |
| g01e01 | g00c03 | -2.4734 |

## Generation 2

| id | parent | native score |
|---|---|---:|
| g02c03 | g01c02 | 5.1258 |
| g02e00 | g01c03 | 3.5500 |
| g02c02 | g01c03 | 3.3769 |
| g02e01 | g01c02 | -1.3397 |

## Generation 3

| id | parent | native score |
|---|---|---:|
| g03e00 | g02c03 | 5.1258 |
| g03e01 | g02e00 | 3.5500 |
| g03c02 | g02c03 | 3.4548 |
| g03c03 | g02e00 | 2.4186 |

## Best reward

`runs\env_007\fragilecargo_eureka\seed_0\gen_02\cand_03\reward_v1.py`

copied to `runs\env_007\fragilecargo_eureka\seed_0\best\best_reward.py`