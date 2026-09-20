def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    components = {}

    # ---------- 1) 主进展：货箱到 dock 距离的净减少（米） ----------
    dist_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    dist_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    raw_progress = dist_prev - dist_next
    if raw_progress > 0.15:
        raw_progress = 0.15
    if raw_progress < -0.15:
        raw_progress = -0.15
    components["crate_progress_toward_dock"] = 2.0 * raw_progress

    # ---------- 2) 停靠沉降：距离 + 近静止 + 朝向对齐 的连续联合代理 ----------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    f_dist = 0.0
    if dist_next < 0.8:
        f_dist = 1.0 - (dist_next / 0.8) ** 2

    f_speed = 0.0
    if crate_speed < 0.3:
        f_speed = 1.0 - crate_speed / 0.3

    aligned = next_obs[10] ** 2 - next_obs[11] ** 2
    if aligned < 0.0:
        aligned = -aligned
    f_align = 0.0
    if aligned > 0.5:
        f_align = (aligned - 0.5) / 0.5

    joint = (f_dist * f_speed * f_align) ** (1.0 / 3.0)
    components["docking_settle_success"] = 0.8 * joint

    # ---------- 3) 脆弱搬运保护：初次接触瞬间的相对接近速度做 hinge 惩罚 ----------
    r_fragile = 0.0
    if next_obs[14] > 0.5 and obs[14] < 0.5:
        cart_vx = obs[4] * 3.0 * obs[2]
        cart_vy = obs[4] * 3.0 * obs[3]
        rel_vx = cart_vx - obs[8] * 3.0
        rel_vy = cart_vy - obs[9] * 3.0
        rel_speed = (rel_vx * rel_vx + rel_vy * rel_vy) ** 0.5
        over = rel_speed - 0.5
        if over < 0.0:
            over = 0.0
        if over > 1.0:
            over = 1.0
        r_fragile = -0.4 * over
    components["fragile_contact_guard"] = r_fragile

    # ---------- 4) 持续推挤的强度抑制：接触中货箱速度过大时轻罚 ----------
    r_push = 0.0
    if next_obs[14] > 0.5 and crate_speed > 1.0:
        over = crate_speed - 1.0
        if over > 1.0:
            over = 1.0
        r_push = -0.15 * over
    components["push_intensity_guard"] = r_push

    # ---------- 5) 地板边界保护：小车与货箱都靠近/越过地板时罚 ----------
    cart_over = 0.0
    if abs(next_obs[0]) > 0.9:
        cart_over += abs(next_obs[0]) - 0.9
    if abs(next_obs[1]) > 0.9:
        cart_over += abs(next_obs[1]) - 0.9

    rx = next_obs[6] * 3.0
    ry = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + next_obs[2] * rx - next_obs[3] * ry
    crate_wy = next_obs[1] * 4.0 + next_obs[3] * rx + next_obs[2] * ry
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0

    crate_over = 0.0
    if abs(crate_nx) > 0.9:
        crate_over += abs(crate_nx) - 0.9
    if abs(crate_ny) > 0.9:
        crate_over += abs(crate_ny) - 0.9

    components["floor_bounds_guard"] = -1.0 * (cart_over + crate_over)

    # ---------- 6) 轻量动作代价：抑制无效转向抖动 ----------
    components["steering_effort"] = -0.01 * (action[1] ** 2)

    total = 0.0
    for key in components:
        total += components[key]

    return (float(total), components)