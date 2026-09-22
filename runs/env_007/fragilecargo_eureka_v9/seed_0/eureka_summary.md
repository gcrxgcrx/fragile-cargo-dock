# EUREKA-style population search - summary

- environment: `FragileCargoDock-v0`
- generations: 4, population 4 (elite 2 x 1 + 1 children each)
- timesteps per candidate: 3,000,000
- evaluation episodes: 20
- selection metric: native `external_eval.mean_eval_reward`
- seed: 0

**Best candidate: `g03c03` with native score 278.7136**

## Generation 0

| id | parent | native score |
|---|---|---:|
| g00c02 | - | 6.6726 |
| g00c00 | - | 3.9019 |
| g00c03 | - | 2.0813 |
| g00c01 | - | 0.1779 |

## Generation 1

| id | parent | native score |
|---|---|---:|
| g01e00 | g00c02 | 6.6726 |
| g01e01 | g00c00 | 3.9019 |
| g01c03 | g00c00 | 3.6098 |
| g01c02 | g00c02 | 3.5296 |

## Generation 2

| id | parent | native score |
|---|---|---:|
| g02e00 | g01e00 | 6.6726 |
| g02e01 | g01e01 | 3.9019 |
| g02c03 | g01e01 | -0.8217 |
| g02c02 | g01e00 | -1.7789 |

## Generation 3

| id | parent | native score |
|---|---|---:|
| g03c03 | g02e01 | 278.7136 |
| g03e00 | g02e00 | 6.6726 |
| g03e01 | g02e01 | 3.9019 |
| g03c02 | g02e00 | -2.1072 |

## Best reward

`runs\env_007\fragilecargo_eureka_v9\seed_0\gen_03\cand_03\reward_v1.py`

copied to `runs\env_007\fragilecargo_eureka_v9\seed_0\best\best_reward.py`