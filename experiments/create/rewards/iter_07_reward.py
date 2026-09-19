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

    s_front = obs[15]
    s_left = obs[16]
    s_right = obs[17]

    # ---------- A. MAIN: crate_to_dock_progress (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    # 主信号：对称，靠近给正分，远离给负分
    w_progress = 30.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (joint_condition_proxy, 几何平均) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.5)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重压低，避免占据状态即刷分
    w_joint = 0.4
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. local_obstacle_penalty (hinge, 局部障碍接近度) ----------
    # 只在接近度超过 0.7 时惩罚，避免全时惩罚压制探索
    obs_excess = max(0.0, s_front - 0.7) + max(0.0, s_left - 0.7) + max(0.0, s_right - 0.7)
    w_obs = 3.0
    r_obs = -w_obs * obs_excess

    # ---------- total ----------
    total_reward = r_progress + r_joint + r_impact + r_obs

    components = {
        "crate_to_dock_progress": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "local_obstacle_penalty": r_obs,
    }

    return float(total_reward), components