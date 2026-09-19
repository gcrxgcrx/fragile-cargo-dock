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

    # ---------- A. MAIN: dock_progress_delta (improvement_delta) ----------
    # 货箱到 dock 距离的逐步减少量，唯一主学习信号
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 40.0
    r_progress = w_progress * progress

    # ---------- B. GATE: joint_completion_gate (几何平均, 无 floor 塌缩) ----------
    # 三因子：near / slow / align，连续 bounded，几何平均防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)

    # 几何平均：任一因子为 0 时整体为 0，但用 0.1 下限避免完全塌缩
    f_near_c = max(f_near, 0.1)
    f_slow_c = max(f_slow, 0.1)
    f_align_c = max(f_align, 0.1)
    gate = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 门控乘子：完成度越高，主信号越被放大（不独立给分）
    w_gate = 15.0
    r_gate = w_gate * gate * max(0.0, progress)

    # ---------- C. crate_speed_hinge (停稳引导, 只在货箱过快时罚) ----------
    speed_excess = max(0.0, crate_speed - 0.15)
    w_speed = 4.0
    r_speed = -w_speed * speed_excess

    # ---------- D. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.25)
    w_impact = 2.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- E. local_obstacle_penalty (hinge, 基于 obs[15..17]) ----------
    # 只在接近静态障碍时罚，给探索留空间
    obs_front = obs[15]
    obs_left = obs[16]
    obs_right = obs[17]
    obs_max = max(obs_front, obs_left, obs_right)
    obstacle_excess = max(0.0, obs_max - 0.7)
    w_obstacle = 2.0
    r_obstacle = -w_obstacle * obstacle_excess

    # ---------- total ----------
    total_reward = r_progress + r_gate + r_speed + r_impact + r_obstacle

    components = {
        "dock_progress_delta": r_progress,
        "joint_completion_gate": r_gate,
        "crate_speed_hinge": r_speed,
        "fragile_impact_penalty": r_impact,
        "local_obstacle_penalty": r_obstacle,
    }

    return float(total_reward), components