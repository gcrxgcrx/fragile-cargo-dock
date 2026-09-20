def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    components = {}

    # ---------- observable geometry ----------
    c = next_obs[2]
    s = next_obs[3]
    bx = next_obs[6] * 3.0
    by = next_obs[7] * 3.0
    crate_wx = c * bx - s * by + next_obs[0] * 5.0
    crate_wy = s * bx + c * by + next_obs[1] * 4.0

    d_now = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    d_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5

    # ---------- 1) main dense signal: net crate progress toward the dock ----------
    delta = d_now - d_next
    if delta > 0.2:
        delta = 0.2
    if delta < -0.2:
        delta = -0.2
    components["crate_dock_progress"] = 6.0 * delta

    # ---------- 2) joint settle proxy: near dock + slow + axis aligned ----------
    f_dist = 1.0 - d_next / 1.0
    if f_dist < 0.0:
        f_dist = 0.0

    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    f_speed = 1.0 - crate_speed / 0.5
    if f_speed < 0.0:
        f_speed = 0.0

    cc = next_obs[10]
    ss = next_obs[11]
    if abs(cc) < abs(ss):
        ang_err = abs(cc)
    else:
        ang_err = abs(ss)
    f_ang = 1.0 - ang_err / 0.7
    if f_ang < 0.0:
        f_ang = 0.0

    if f_dist > 0.0 and f_speed > 0.0 and f_ang > 0.0:
        settle = (f_dist * f_speed * f_ang) ** (1.0 / 3.0)
    else:
        settle = 0.0
    components["dock_settle_proxy"] = 1.5 * settle

    # ---------- 3) fragile handling guard: penalize hard contact approach speed ----------
    vx_crate = obs[8] * 3.0
    vy_crate = obs[9] * 3.0
    rel_vx = (c * vx_crate + s * vy_crate) - obs[4] * 3.0
    rel_vy = (-s * vx_crate + c * vy_crate)
    rel_speed = (rel_vx ** 2 + rel_vy ** 2) ** 0.5
    excess = rel_speed - 0.6
    if excess < 0.0:
        excess = 0.0
    contact = 0.0
    if next_obs[14] > 0.5:
        contact = 1.0
    components["fragile_impact_guard"] = -1.5 * contact * excess * excess

    # ---------- 4) soft bounds guards ----------
    over_cx = abs(crate_wx) - 4.7
    if over_cx < 0.0:
        over_cx = 0.0
    over_cy = abs(crate_wy) - 3.7
    if over_cy < 0.0:
        over_cy = 0.0
    components["crate_bounds_guard"] = -4.0 * (over_cx + over_cy)

    over_rx = abs(next_obs[0]) - 0.9
    if over_rx < 0.0:
        over_rx = 0.0
    over_ry = abs(next_obs[1]) - 0.9
    if over_ry < 0.0:
        over_ry = 0.0
    components["cart_bounds_guard"] = -4.0 * (over_rx + over_ry)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)