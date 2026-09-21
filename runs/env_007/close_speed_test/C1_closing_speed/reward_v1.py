# C1_closing_speed — built by make_closing_speed_variants.py
#
# Base        : runs\env_007\control_obs_only\reward.py
# Edit        : added the observation-only closing-speed penalty (-0.05 * contact * closing_speed); nothing else
# Everything else (both telescoping shaping terms, the +20 settled hold, the boundary
# guard, the shove penalty, every coefficient) is control_v1's own code, unmodified.
# Protocol and predictions: runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md

"""CONTROL: a hand-written, observation-only reward for FragileCargoDock-v0.

Why this file exists
--------------------
Every reward produced by the LLM search uses only observations, so the search's
ceiling is bounded by what the observation contract can express under the
training harness (`reward_clip = 20`, `normalize_reward = true`,
`gamma = 0.999`). That ceiling has never been measured: the 96.8 % reference is
the *native* reward, which bypasses the wrapper entirely and is neither clipped
nor normalised, so it proves nothing about this contract.

This control is hand-written to satisfy every mechanical check in
`analyze_terminal_dominance.py`, and then trained with exactly the same
hyperparameters as the search candidates.

Design rules it obeys
---------------------
* Every shaping term is an **increment** (`f(obs) - f(next_obs)`), so a crate
  that is stationary anywhere - including 0.13 m from the dock - earns exactly
  zero per step. There is no persistent near-goal state reward to farm.
* The completion condition is taken verbatim from the task spec:
  `|obs[12]| <= 0.024`, `|obs[13]| <= 0.030`, `obs[10] >= cos(30 deg)`,
  crate speed < 0.05 m/s.
* The completion term is worth the clip maximum for every one of the ten
  settling steps the environment requires, so a completion is worth
  `10 * 20 = 200` while the whole dense path is worth about `4.1` (the initial
  crate-dock distance), a ratio of ~49:1.
* No term can be farmed by standing still, and no term is negative while the
  crate is being pushed toward the dock, so "give up and end the episode early"
  is never attractive - which is what drove cand_00 and cand_02 out of bounds.

The two shaping terms mirror the environment's own potential-based terms
(robot-to-crate approach, crate-to-dock progress), which are both exactly
recoverable from the observation:

    crate-to-dock distance  = hypot(obs[12] * 5.0, obs[13] * 4.0)
    cart-to-crate distance  = hypot(obs[6] * 3.0, obs[7] * 3.0)
"""


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- crate -> dock distance, in metres (incremental) ------------------
    dx0, dy0 = obs[12] * 5.0, obs[13] * 4.0
    dx1, dy1 = next_obs[12] * 5.0, next_obs[13] * 4.0
    dist_prev = (dx0 * dx0 + dy0 * dy0) ** 0.5
    dist_now = (dx1 * dx1 + dy1 * dy1) ** 0.5
    crate_progress = dist_prev - dist_now

    # ---- cart -> crate distance, in metres (incremental) ------------------
    rx0, ry0 = obs[6] * 3.0, obs[7] * 3.0
    rx1, ry1 = next_obs[6] * 3.0, next_obs[7] * 3.0
    gap_prev = (rx0 * rx0 + ry0 * ry0) ** 0.5
    gap_now = (rx1 * rx1 + ry1 * ry1) ** 0.5
    cart_approach = gap_prev - gap_now

    # ---- completion condition, exactly as the task spec states it ---------
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    inside_x = abs(next_obs[12]) <= 0.024
    inside_y = abs(next_obs[13]) <= 0.030
    aligned = next_obs[10] >= 0.8660254037844387      # cos(30 degrees)
    slow = crate_speed < 0.05

    # The environment terminates the episode on success, so this can be
    # collected at most STABLE_STEPS_REQUIRED = 10 times.
    settled_hold = 20.0 if (inside_x and inside_y and aligned and slow) else 0.0

    # ---- keep the cart on the floor --------------------------------------
    # The cart leaves the field past |obs[0]| = 1.05 (5.25 m / 5.0 m) and
    # |obs[1]| = 1.05, so hold it back from 0.95.
    edge = abs(next_obs[0])
    if abs(next_obs[1]) > edge:
        edge = abs(next_obs[1])
    boundary = -5.0 * (edge - 0.95) if edge > 0.95 else 0.0

    # ---- only genuinely violent shoves, far above any useful pushing speed -
    shove = 0.0
    if next_obs[14] > 0.5 and crate_speed > 1.5:
        shove = -0.2 * (crate_speed - 1.5)

    # ---- the ONE added term: an observation-only closing-speed penalty ------
    # Copied verbatim from ablation_probe/probeD_obs_gentleness.py. This is the
    # only difference between C0 and C1.
    cos_h = obs[2]
    sin_h = obs[3]
    cart_forward_speed = obs[4] * 3.0
    crate_speed_along_heading = cvx * cos_h + cvy * sin_h
    closing_speed = cart_forward_speed - crate_speed_along_heading
    if closing_speed < 0.0:
        closing_speed = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing_speed

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": float(settled_hold),
        "boundary": float(boundary),
        "shove": float(shove),
        "gentleness_obs": float(gentleness),
    }
    total = (crate_progress + cart_approach + settled_hold + boundary + shove + gentleness)
    return float(total), components
