def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]

    cart_fwd = obs[4]
    contact = obs[14]

    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- A. MAIN: dock_approach_improvement (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标下）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new

    # 联合完成门控：接近 + 停稳 + 对齐，几何平均，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)
    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    gate = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 主信号 = 改善量 × 门控（门控在 0.05~1 之间，不塌缩）
    w_progress = 20.0
    r_progress = w_progress * progress * gate

    # ---------- B. AUX: dock_completion_state (bounded state, 低权重) ----------
    # 保留一个低权重的状态信号，避免货箱已到位但无梯度时完全无反馈
    w_state = 1.5
    r_state = w_state * gate

    # ---------- C. fragile_impact_penalty (hinge, only on contact) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. boundary_health_penalty (hinge) ----------
    bx = max(0.0, abs(cart_x) - 0.9)
    by = max(0.0, abs(cart_y) - 0.9)
    w_bounds = 6.0
    r_bounds = -w_bounds * (bx + by)

    # ---------- total ----------
    total_reward = r_progress + r_state + r_impact + r_bounds

    components = {
        "dock_approach_improvement": r_progress,
        "dock_completion_state": r_state,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components