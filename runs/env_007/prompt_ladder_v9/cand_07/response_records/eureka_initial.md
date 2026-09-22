# Response Record

```python
# ---------------------------------------------------------------------------
# Module level state (MUST be declared before compute_reward; no `global` used)
# ---------------------------------------------------------------------------
_PREV_T = [-1.0]        # obs[18] of the previous call -> episode-boundary detector
_STREAK = [0]           # consecutive steps with the full docking predicate true
_PAID = [False]         # one-shot terminal_success already paid this episode
_ENTERED = [False]      # crate has already been fully inside the dock once
_HARD_HITS = [0]        # self-counted hard impacts (impulse channel does not exist)
_FAILED = [False]       # one-shot terminal_failure already paid this episode

# geometry / thresholds, all taken from the environment facts
_HALF_W = 5.0           # warehouse half width  [m]  (obs[0], obs[12] scale)
_HALF_H = 4.0           # warehouse half height [m]  (obs[1], obs[13] scale)
_DOCK_TOL_X = 0.024     # |obs[12]| <= this  -> crate fully inside the dock
_DOCK_TOL_Y = 0.030     # |obs[13]| <= this  -> crate fully inside the dock
_ALIGN_COS = 0.866      # cos(30 deg) heading-alignment tolerance
_SPEED_TOL = 0.05       # [m/s] crate must be (almost) at rest
_SUCCEED_STEPS = 10     # consecutive settled steps that end the episode
_ROUGH_K = 0.25         # gentleness gain (see self-check 3)
_HARD_CLOSING = 1.0     # [m/s] closing speed above which a contact is a "hard hit"
_SETTLE_STEP_REWARD = 2.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------------
    # SELF-CHECK RECORD (per-step values, before the environment clip at 20)
    #   R_idle    (no motion, crate frozen)        ~ -0.002  (time cost only)
    #   R_push    (steady push toward the dock)    ~ +0.05 (progress, 1 m/s * dt)
    #                                                - 0.002 - ~0.001 (action)
    #                                                - roughness ~ 0 (steady
    #                                                  pushing has closing ~ 0)
    #                                                ~ +0.047
    #   R_settled (in dock + aligned + slow)       ~ +2.0 - 0.002 = +1.998
    #  (1) R_push > R_idle and the gap ~0.049 is far above the largest single
    #      step penalty (time 0.002 / action 0.001 / roughness during normal
    #      pushing ~0.005), so acting is never worse than standing still.
    #  (2) R_settled > R_push  -> settling is the unique global optimum.
    #  (3) 400 hovering steps outside the dock: ~ -0.8 total
    #      10 settled steps + dock_enter:     ~ +25 total  -> (2) check passed.
    #  (4) gentleness at closing = 1.0 vs 0.05 m/s differs by 0.25*0.95 = 0.238,
    #      i.e. several times the normal per-step push term -> impact cannot pay.
    #  (5) same settled state called 12x in a row: the per-step delta stays
    #      constant and positive (settle_reward is never switched off).
    # ------------------------------------------------------------------

    # ---- 0. episode boundary detection (obs[18] is monotone inside an episode)
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED[0] = False
    _PREV_T[0] = t

    # ---- 1. approach_cargo : cart -> crate distance closed this frame (+1/m)
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    nrel_x = next_obs[6] * 3.0
    nrel_y = next_obs[7] * 3.0
    d_cc_prev = (rel_x * rel_x + rel_y * rel_y) ** 0.5
    d_cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5
    approach_cargo = 1.0 * (d_cc_prev - d_cc_next)          # signed, symmetric

    # ---- 2. progress : crate -> dock distance closed this frame (+1/m)
    dx_prev = obs[12] * _HALF_W
    dy_prev = obs[13] * _HALF_H
    dx_next = next_obs[12] * _HALF_W
    dy_next = next_obs[13] * _HALF_H
    d_dock_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5
    d_dock_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5
    progress = 1.0 * (d_dock_prev - d_dock_next)            # signed, symmetric

    # ---- 3. contact + closing-speed proxy (no impulse channel exists)
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0

    # ---- 4. roughness : contact-impulse proxy ("handle the crate gently")
    #         equal 0 while pushing steadily (closing ~ 0), strongly negative
    #         only for a fast impact -> suppresses ramming, not propulsion.
    roughness = -_ROUGH_K * contact * closing

    # ---- 5. hard hit (single-step fixed penalty) + self-counted impulse count
    hard_hit = 0.0
    if contact > 0.5 and closing > _HARD_CLOSING:
        _HARD_HITS[0] += 1
        hard_hit = -0.5

    # ---- 6. docking predicate (all thresholds from the environment facts)
    in_dock = (abs(next_obs[12]) <= _DOCK_TOL_X) and (abs(next_obs[13]) <= _DOCK_TOL_Y)
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    aligned = next_obs[10] >= _ALIGN_COS
    slow = crate_speed < _SPEED_TOL
    settled = in_dock and aligned and slow

    # first time the crate is completely inside the dock -> one-shot +5
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # consecutive settled steps -> one-shot +300 (episode ends at 10 anyway)
    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= _SUCCEED_STEPS and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # per-step revenue while the docking predicate holds.  It is a GATED
    # (in-dock + aligned + slow) signal, never a global state bonus, and it is
    # deliberately NOT switched off by the one-shot event above.
    settle_reward = _SETTLE_STEP_REWARD if settled else 0.0

    # ---- 7. out-of-bounds guard: cart (obs[0], obs[1]) and crate (derived)
    ax = abs(obs[0])
    ay = abs(obs[1])
    cart_edge = ax if ax > ay else ay
    boundary_penalty = 0.0
    if cart_edge > 0.92:
        boundary_penalty = -6.0 * (cart_edge - 0.92) / 0.13
        if boundary_penalty < -8.0:
            boundary_penalty = -8.0

    cos_h = obs[2]
    sin_h = obs[3]
    crate_wx = obs[0] * _HALF_W + rel_x * cos_h - rel_y * sin_h
    crate_wy = obs[1] * _HALF_H + rel_x * sin_h + rel_y * cos_h
    crate_ex = abs(crate_wx) / _HALF_W
    crate_ey = abs(crate_wy) / _HALF_H
    crate_edge = crate_ex if crate_ex > crate_ey else crate_ey
    if crate_edge > 0.98:
        crate_pen = -5.0 * (crate_edge - 0.98) / 0.07
        if crate_pen < -8.0:
            crate_pen = -8.0
        boundary_penalty += crate_pen

    # ---- 8. action cost + fixed time cost
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---- 9. one-shot terminal failure (out of bounds, or >= 3 hard hits)
    terminal_failure = 0.0
    if not _FAILED[0]:
        if cart_edge > 1.05 or crate_edge > 1.05 or _HARD_HITS[0] >= 3:
            _FAILED[0] = True
            terminal_failure = -100.0

    total_reward = (approach_cargo
                    + progress
                    + dock_enter
                    + roughness
                    + action_cost
                    + time_cost
                    + hard_hit
                    + settle_reward
                    + boundary_penalty
                    + terminal_success
                    + terminal_failure)

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settle_reward": settle_reward,
        "boundary_penalty": boundary_penalty,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }
    return float(total_reward), components
```
