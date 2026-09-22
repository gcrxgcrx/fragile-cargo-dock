def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    if not hasattr(compute_reward, "_st"):
        compute_reward._st = {
            "prev_t": -1.0,
            "hold": 0,
            "dwell": 0,
            "hard": 0,
            "guard": False,
            "paid": False,
        }
    st = compute_reward._st

    t = float(next_obs[18])
    if t < st["prev_t"] or t <= 1.0 / 400.0:
        st["hold"] = 0
        st["dwell"] = 0
        st["hard"] = 0
        st["guard"] = False
        st["paid"] = False
    st["prev_t"] = t

    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    ch = obs[2]
    sh = obs[3]
    bx = obs[6] * 3.0
    by = obs[7] * 3.0
    crate_x = cart_x + bx * ch - by * sh
    crate_y = cart_y + bx * sh + by * ch

    n_cart_x = next_obs[0] * 5.0
    n_cart_y = next_obs[1] * 4.0
    n_ch = next_obs[2]
    n_sh = next_obs[3]
    n_bx = next_obs[6] * 3.0
    n_by = next_obs[7] * 3.0
    n_crate_x = n_cart_x + n_bx * n_ch - n_by * n_sh
    n_crate_y = n_cart_y + n_bx * n_sh + n_by * n_ch

    d_cc_prev = ((crate_x - cart_x) ** 2 + (crate_y - cart_y) ** 2) ** 0.5
    d_cc_next = ((n_crate_x - n_cart_x) ** 2 + (n_crate_y - n_cart_y) ** 2) ** 0.5
    approach_cargo = (d_cc_prev - d_cc_next) * 0.6

    dk_px = obs[12] * 5.0
    dk_py = obs[13] * 4.0
    dk_nx = next_obs[12] * 5.0
    dk_ny = next_obs[13] * 4.0
    d_dock_prev = (dk_px ** 2 + dk_py ** 2) ** 0.5
    d_dock_next = (dk_nx ** 2 + dk_ny ** 2) ** 0.5
    progress = (d_dock_prev - d_dock_next) * 1.5

    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5
    crate_along = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    soft = closing - 0.20
    if soft < 0.0:
        soft = 0.0
    harsh = closing - 0.70
    if harsh < 0.0:
        harsh = 0.0
    roughness = -0.10 * contact * soft - 0.30 * contact * harsh

    hard_hit = 0.0
    if contact > 0.5 and closing > 1.00:
        hard_hit = -0.5
        st["hard"] = st["hard"] + 1

    axis_align = abs(next_obs[10])
    if abs(next_obs[11]) > axis_align:
        axis_align = abs(next_obs[11])

    near = 1.0 - d_dock_next / 0.45
    if near < 0.0:
        near = 0.0
    slowf = 1.0 / (1.0 + 10.0 * crate_speed)
    dock_bonus = 0.35 * (near * near) * slowf * axis_align

    in_box = (abs(dk_nx) < 0.22) and (abs(dk_ny) < 0.22)
    slow = crate_speed < 0.06
    settling = in_box and slow

    dwell = 0.0
    if settling and st["dwell"] < 30:
        st["dwell"] = st["dwell"] + 1
        dwell = 0.5 * (0.5 + 0.5 * axis_align)

    if settling and axis_align >= 0.866:
        st["hold"] = st["hold"] + 1
    else:
        st["hold"] = 0

    dock_hold_bonus = 0.0
    if st["hold"] >= 10 and not st["paid"]:
        st["paid"] = True
        dock_hold_bonus = 15.0

    cart_edge = abs(next_obs[0])
    if abs(next_obs[1]) > cart_edge:
        cart_edge = abs(next_obs[1])
    crate_edge = abs(n_crate_x) / 5.0
    if abs(n_crate_y) / 4.0 > crate_edge:
        crate_edge = abs(n_crate_y) / 4.0
    edge = cart_edge
    if crate_edge > edge:
        edge = crate_edge

    bounds_penalty = 0.0
    if edge > 0.92:
        bounds_penalty = bounds_penalty - 2.0 * (edge - 0.92)
    if edge > 1.00:
        bounds_penalty = bounds_penalty - 6.0 * (edge - 1.00)
    if bounds_penalty < -6.0:
        bounds_penalty = -6.0

    cart_oob = (abs(next_obs[0]) > 1.02) or (abs(next_obs[1]) > 1.02)
    crate_oob = (abs(n_crate_x) > 5.10) or (abs(n_crate_y) > 4.08)
    safety_guard = 0.0
    if (cart_oob or crate_oob or st["hard"] >= 3) and not st["guard"]:
        st["guard"] = True
        safety_guard = -25.0

    action_cost = -0.0004 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.001

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_bonus": dock_bonus,
        "dwell": dwell,
        "dock_hold_bonus": dock_hold_bonus,
        "roughness": roughness,
        "hard_hit": hard_hit,
        "bounds_penalty": bounds_penalty,
        "safety_guard": safety_guard,
        "action_cost": action_cost,
        "time_cost": time_cost,
    }
    total = 0.0
    for key in components:
        total = total + components[key]
    return float(total), components