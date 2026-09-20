_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode boundary detection ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- geometry ----------
    # warehouse half extents
    HALF_W = 5.0
    HALF_H = 4.0

    # crate-to-dock offsets (normalized)
    dnx = float(next_obs[12])
    dny = float(next_obs[13])
    dx_m = dnx * HALF_W
    dy_m = dny * HALF_H
    dist = (dx_m * dx_m + dy_m * dy_m) ** 0.5

    # previous distance for delta signal
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    prev_dist = _PREV_DIST[0]

    # ---------- dock tolerance / completion condition ----------
    in_dock = (abs(dnx) <= 0.024) and (abs(dny) <= 0.030)

    # crate heading error
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # angle error relative to dock axis (assume dock axis = +x, cos=1)
    # error = atan2(sin, cos); use cos of angle as alignment proxy
    align = crate_cos  # in [-1, 1], 1 = fully aligned
    if align < -1.0:
        align = -1.0
    if align > 1.0:
        align = 1.0
    aligned = align >= 0.866  # < 30 deg

    # crate speed
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = crate_speed < 0.05

    # completion condition
    complete = in_dock and aligned and slow

    # streak counting
    if complete:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- components ----------
    components = {}

    # 1) progress delta (incremental, only when closer)
    progress = prev_dist - dist  # positive if closer
    if progress > 0.0:
        progress_reward = 2.0 * progress
    else:
        progress_reward = 0.0
    components["progress"] = progress_reward

    # 2) alignment-gated progress bonus (small)
    align_gate = (align + 1.0) * 0.5  # [0,1]
    if progress > 0.0:
        components["align_progress"] = 0.5 * progress * align_gate
    else:
        components["align_progress"] = 0.0

    # 3) gentleness penalty (soft contact)
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    cart_cos = float(obs[2])
    cart_sin = float(obs[3])
    crate_along_heading = crate_vx * cart_cos + crate_vy * cart_sin
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -2.0 * contact * closing
    components["gentleness"] = gentleness

    # 4) speed penalty near dock (only when close and moving fast)
    near_dock = (abs(dnx) <= 0.15) and (abs(dny) <= 0.15)
    if near_dock and not complete:
        overspeed = crate_speed - 0.05
        if overspeed > 0.0:
            components["dock_speed_penalty"] = -1.0 * overspeed
        else:
            components["dock_speed_penalty"] = 0.0
    else:
        components["dock_speed_penalty"] = 0.0

    # 5) out-of-bounds guard for cart
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    bound_pen = 0.0
    if cx > 0.95:
        bound_pen -= 5.0 * (cx - 0.95)
    if cy > 0.95:
        bound_pen -= 5.0 * (cy - 0.95)
    components["bounds"] = bound_pen

    # 6) out-of-bounds guard for crate (via dock offset + cart pos, approximate)
    # crate world pos approx: cart_pos + rotated rel
    relx = float(next_obs[6]) * 3.0
    rely = float(next_obs[7]) * 3.0
    crate_wx = float(next_obs[0]) * HALF_W + relx * cart_cos - rely * cart_sin
    crate_wy = float(next_obs[1]) * HALF_H + relx * cart_sin + rely * cart_cos
    crate_bound_x = abs(crate_wx) / HALF_W
    crate_bound_y = abs(crate_wy) / HALF_H
    crate_bound_pen = 0.0
    if crate_bound_x > 0.95:
        crate_bound_pen -= 5.0 * (crate_bound_x - 0.95)
    if crate_bound_y > 0.95:
        crate_bound_pen -= 5.0 * (crate_bound_y - 0.95)
    components["crate_bounds"] = crate_bound_pen

    # 7) one-time entry bonus
    entry_bonus = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_bonus = 20.0
    components["entry_bonus"] = entry_bonus

    # 8) one-time success event
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- completion freeze ----------
    # When in completed state (in_dock + aligned + slow), zero out all
    # persistent components; only one-time events may remain.
    if complete and success_event == 0.0 and entry_bonus == 0.0:
        components["progress"] = 0.0
        components["align_progress"] = 0.0
        components["gentleness"] = 0.0
        components["dock_speed_penalty"] = 0.0
        components["bounds"] = 0.0
        components["crate_bounds"] = 0.0

    # update prev distance
    _PREV_DIST[0] = dist

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components