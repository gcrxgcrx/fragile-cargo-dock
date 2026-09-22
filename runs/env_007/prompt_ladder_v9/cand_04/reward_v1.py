_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode boundary detection (obs[18] is monotone within an episode) ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    # ---- 1. approach_cargo: cart->crate distance decrease, meters ----
    prev_cart_crate = 3.0 * ((obs[6] * obs[6] + obs[7] * obs[7]) ** 0.5)
    next_cart_crate = 3.0 * ((next_obs[6] * next_obs[6] + next_obs[7] * next_obs[7]) ** 0.5)
    approach_cargo = prev_cart_crate - next_cart_crate

    # ---- 2. progress: crate->dock distance decrease, meters ----
    prev_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    next_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = prev_dock - next_dock

    # ---- 3. dock_enter: first time crate fully inside dock tolerance ----
    dock_enter = 0.0
    if abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030:
        if not _ENTERED[0]:
            _ENTERED[0] = True
            dock_enter = 5.0

    # ---- 4. roughness: observable proxy for contact impulse ----
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.05 * contact * closing

    # ---- 5. action_cost ----
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])

    # ---- 6. time_cost ----
    time_cost = -0.002

    # ---- 7. hard_hit: single-step hard impact proxy ----
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # ---- 8. terminal_success: 10 consecutive steps of completion predicate ----
    inside_dock = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030)
    aligned = next_obs[10] >= 0.866025  # cos(30 deg) for heading aligned with x-axis
    speed_sq = (next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2
    slow = speed_sq < (0.05 ** 2)
    if inside_dock and aligned and slow:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # ---- 9. terminal_failure: out-of-bounds or >=3 hard hits ----
    cart_out = (abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05)
    dx_body = next_obs[6] * 3.0
    dy_body = next_obs[7] * 3.0
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    dx_world = dx_body * cos_h - dy_body * sin_h
    dy_world = dx_body * sin_h + dy_body * cos_h
    x_crate = next_obs[0] * 5.0 + dx_world
    y_crate = next_obs[1] * 4.0 + dy_world
    x_crate_norm = x_crate / 5.0
    y_crate_norm = y_crate / 4.0
    crate_out = (abs(x_crate_norm) > 1.05 or abs(y_crate_norm) > 1.05)
    hard_hits_out = _HARD_HITS[0] >= 3
    terminal_failure = -100.0 if (cart_out or crate_out or hard_hits_out) else 0.0

    # ---- 10. boundary guard: monotone penalty near cart walls ----
    bx = abs(next_obs[0]) - 0.95
    if bx < 0.0:
        bx = 0.0
    by = abs(next_obs[1]) - 0.95
    if by < 0.0:
        by = 0.0
    boundary_penalty = -10.0 * (bx + by)

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "boundary_penalty": boundary_penalty,
    }
    total = (approach_cargo + progress + dock_enter + roughness +
             action_cost + time_cost + hard_hit + terminal_success +
             terminal_failure + boundary_penalty)
    return float(total), components