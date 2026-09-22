```python
# Module-level state for one-time events and episode-boundary detection.
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_FAILED = [False]
_HARD_HITS = [0]

# Self-check notes (monetary units are reward units):
# ① idle ~ -0.002; normal push ~ +0.05; push > idle.
# ② hovering outside dock 0.3 m for 400 steps ~ -0.8; settled inside for 10 steps
#    gives +2.0/step (+20) plus terminal_success +300 once; settled > hover.
# ③ closing 1.0 m/s: roughness -0.05, hard_hit -0.5 => -0.55;
#    closing 0.05 m/s: roughness -0.0025, hard_hit 0 => -0.0025;
#    difference 0.5475 > normal progress per step ~0.05.
# ④ same settled state: settle_step = +2.0 every call, so cumulative grows linearly.
# ⑤ R_idle ~ -0.002, R_push ~ +0.05, R_settled ~ +2.0.
#    R_push > R_idle and R_settled > R_push.
# Boundary: center reward ~ -0.002; |x| = 1.05 gives boundary_guard -20 and
# terminal_failure -100 once.

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------------------------------------------------------
    # Episode boundary detection using monotonically increasing obs[18].
    # ---------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # ---------------------------------------------------------------
    # Geometry from declared observation indices.
    # ---------------------------------------------------------------
    # Cart -> crate distance in meters (body-frame relative vector scaled by 3.0).
    d_cc_prev = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    d_cc_next = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = d_cc_prev - d_cc_next

    # Crate -> dock distance in meters (crate_to_dock offsets scaled by 5.0 and 4.0).
    d_cd_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    d_cd_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = d_cd_prev - d_cd_next

    # ---------------------------------------------------------------
    # dock_enter: first time crate is fully inside dock tolerance.
    # ---------------------------------------------------------------
    inside_now = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if inside_now and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------------------------------------------------------------
    # Observability proxy for contact roughness / gentleness.
    # No impulse is available, so use closing speed along cart heading.
    # ---------------------------------------------------------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.05 * contact * closing

    # ---------------------------------------------------------------
    # Fixed step costs.
    # ---------------------------------------------------------------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------------------------------------------------------------
    # Hard-hit detection: contact plus high closing speed.
    # ---------------------------------------------------------------
    hard = 0.0
    if contact > 0.5 and closing >= 1.0:
        hard = 1.0
        _HARD_HITS[0] += 1
    hard_hit = -0.5 * hard

    # ---------------------------------------------------------------
    # Boundary guard for cart and crate.
    # Cart position is obs[0], obs[1] normalized by half-width/height;
    # +/-1.0 is the wall, > 1.05 is out of bounds.
    # Crate world position is recovered from cart pose plus body-frame offset.
    # ---------------------------------------------------------------
    cart_abs = max(abs(next_obs[0]), abs(next_obs[1]))
    cart_boundary_pen = 0.0
    if cart_abs > 0.95:
        cart_boundary_pen = -20.0 * (cart_abs - 0.95) / 0.10

    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_x_norm = cart_x + (rel_x * cos_h - rel_y * sin_h) / 5.0
    crate_y_norm = cart_y + (rel_x * sin_h + rel_y * cos_h) / 4.0
    crate_abs = max(abs(crate_x_norm), abs(crate_y_norm))
    crate_boundary_pen = 0.0
    if crate_abs > 0.95:
        crate_boundary_pen = -20.0 * (crate_abs - 0.95) / 0.10
    boundary_guard = cart_boundary_pen + crate_boundary_pen

    # ---------------------------------------------------------------
    # Completion predicate: crate fully inside dock, aligned to axes,
    # and slow.  This is the same predicate used by the environment.
    # ---------------------------------------------------------------
    speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align_ok = abs(next_obs[10] * next_obs[10] - next_obs[11] * next_obs[11]) >= 0.5
    settled = inside_now and align_ok and (speed < 0.05)

    if settled:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # One-time completion event.
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # Required per-step settled reward.  It is not switched off after the
    # one-time event; the predicate is what gates it.
    settle_step = 2.0 if settled else 0.0

    # ---------------------------------------------------------------
    # Terminal failure: cart out, crate out, or >= 3 hard hits.
    # ---------------------------------------------------------------
    cart_out = abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05
    crate_out = crate_abs > 1.05
    hard_fail = _HARD_HITS[0] >= 3
    terminal_failure = 0.0
    if not _FAILED[0] and (cart_out or crate_out or hard_fail):
        _FAILED[0] = True
        terminal_failure = -100.0

    # ---------------------------------------------------------------
    # Component dictionary and total reward.
    # ---------------------------------------------------------------
    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "roughness": float(roughness),
        "action_cost": float(action_cost),
        "time_cost": float(time_cost),
        "hard_hit": float(hard_hit),
        "terminal_success": float(terminal_success),
        "terminal_failure": float(terminal_failure),
        "settle_step": float(settle_step),
        "boundary_guard": float(boundary_guard),
    }
    total = sum(components.values())
    return float(total), components
```