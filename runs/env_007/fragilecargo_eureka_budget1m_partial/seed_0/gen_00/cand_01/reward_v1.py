def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    components = {}

    # ---- 反解可观测量（严格使用已声明索引） ----
    crate_dx = obs[12] * 5.0
    crate_dy = obs[13] * 4.0
    dist = (crate_dx * crate_dx + crate_dy * crate_dy) ** 0.5

    n_dx = next_obs[12] * 5.0
    n_dy = next_obs[13] * 4.0
    next_dist = (n_dx * n_dx + n_dy * n_dy) ** 0.5

    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0

    n_vx = next_obs[8] * 3.0
    n_vy = next_obs[9] * 3.0
    next_crate_speed = (n_vx * n_vx + n_vy * n_vy) ** 0.5

    # ---- 组件 1：货箱向 dock 的净进展（主学习信号，用 delta 抑制悬停刷分） ----
    delta = dist - next_dist
    if delta > 2.0:
        delta = 2.0
    elif delta < -2.0:
        delta = -2.0
    components["crate_progress_toward_dock"] = 2.0 * (delta / (1.0 + abs(delta)))

    # ---- 组件 2：停稳联合条件（近 + 慢 + 朝向对齐，软完成近似） ----
    near = 1.0 - next_dist / 1.5
    if near < 0.0:
        near = 0.0
    elif near > 1.0:
        near = 1.0

    speed_factor = 1.0 - next_crate_speed / 0.6
    if speed_factor < 0.0:
        speed_factor = 0.0
    elif speed_factor > 1.0:
        speed_factor = 1.0

    cc = next_obs[10]
    ss = next_obs[11]
    ac = cc if cc >= 0.0 else -cc
    asn = ss if ss >= 0.0 else -ss
    align = ac if ac > asn else asn
    align_factor = (align - 0.70710678) / 0.29289322
    if align_factor < 0.0:
        align_factor = 0.0
    elif align_factor > 1.0:
        align_factor = 1.0

    joint = near * speed_factor * align_factor
    settle = joint ** (1.0 / 3.0) if joint > 0.0 else 0.0
    components["docking_settle_success"] = 3.0 * settle

    # ---- 组件 3：脆弱货箱保护（接触瞬间高速接近 = 硬碰撞代理，hinge 惩罚） ----
    contact = obs[14]
    approach = 0.0
    if contact > 0.5:
        ch = obs[2]
        sh = obs[3]
        rx = obs[6] * 3.0
        ry = obs[7] * 3.0
        wx = rx * ch - ry * sh
        wy = rx * sh + ry * ch
        nrm = (wx * wx + wy * wy) ** 0.5
        if nrm > 1e-6:
            nx_dir = wx / nrm
            ny_dir = wy / nrm
            cart_vx = ch * (obs[4] * 3.0)
            cart_vy = sh * (obs[4] * 3.0)
            rel_vx = cart_vx - crate_vx
            rel_vy = cart_vy - crate_vy
            approach = rel_vx * nx_dir + rel_vy * ny_dir
        if approach < 0.0:
            approach = 0.0
    overspeed = approach - 0.5
    if overspeed < 0.0:
        overspeed = 0.0
    components["fragile_handling_guard"] = -1.5 * (overspeed ** 2)

    # ---- 组件 4：越界防护（小车与货箱都不得离开仓库地板，hinge 只在地板边缘附近生效） ----
    gx = next_obs[0]
    if gx < 0.0:
        gx = -gx
    gy = next_obs[1]
    if gy < 0.0:
        gy = -gy
    cart_over = gx - 0.9
    if cart_over < 0.0:
        cart_over = 0.0
    cart_over_y = gy - 0.9
    if cart_over_y < 0.0:
        cart_over_y = 0.0

    nch = next_obs[2]
    nsh = next_obs[3]
    rxb = next_obs[6] * 3.0
    ryb = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + (rxb * nch - ryb * nsh)
    crate_wy = next_obs[1] * 4.0 + (rxb * nsh + ryb * nch)
    cbx = crate_wx / 5.0
    if cbx < 0.0:
        cbx = -cbx
    cby = crate_wy / 4.0
    if cby < 0.0:
        cby = -cby
    crate_over = cbx - 0.9
    if crate_over < 0.0:
        crate_over = 0.0
    crate_over_y = cby - 0.9
    if crate_over_y < 0.0:
        crate_over_y = 0.0

    components["bounds_guard"] = -0.5 * (cart_over + cart_over_y + crate_over + crate_over_y)

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]
    return (float(total_reward), components)