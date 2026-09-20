def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 泊位几何与阈值（来自环境事实） ----
    # 货箱中心到泊位中心的归一化偏移
    dx = next_obs[12]
    dy = next_obs[13]
    # 完全进入泊位的容差（归一化坐标）
    tol_x = 0.024
    tol_y = 0.030

    # ---- 主信号 1：货箱向泊位推进（增量形式，避免悬停收割） ----
    # 用归一化偏移的欧氏距离作为"到泊位距离"的代理
    prev_dist = (obs[12] ** 2 + obs[13] ** 2) ** 0.5
    curr_dist = (dx ** 2 + dy ** 2) ** 0.5
    progress = prev_dist - curr_dist  # 这一帧更接近泊位则为正
    crate_progress = 6.0 * progress

    # ---- 主信号 2：泊位内质量（位置 + 朝向 + 静止的联合代理） ----
    # 位置因子：仅在容差范围内趋近 1
    pos_x = max(0.0, 1.0 - abs(dx) / tol_x)
    pos_y = max(0.0, 1.0 - abs(dy) / tol_y)
    pos_factor = pos_x * pos_y

    # 朝向因子：货箱朝向误差 < 30° 时为 1，超出后线性衰减到 0（60° 处归零）
    crate_heading = (next_obs[10] ** 2 + next_obs[11] ** 2) ** 0.5
    if crate_heading > 1e-6:
        cos_err = next_obs[10] / crate_heading
    else:
        cos_err = 1.0
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    angle_err = (1.0 - cos_err) ** 0.5 * 1.414213562  # 近似 sin(误差)
    # 30° 对应 sin≈0.5，60° 对应 sin≈0.866
    heading_factor = max(0.0, 1.0 - angle_err / 0.866)

    # 静止因子：货箱速度 < 0.05 m/s 时为 1，0.15 m/s 处归零
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    if crate_speed < 0.05:
        speed_factor = 1.0
    elif crate_speed < 0.15:
        speed_factor = (0.15 - crate_speed) / 0.10
    else:
        speed_factor = 0.0

    # 联合质量：几何平均，避免乘积塌缩
    dock_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 4.0 * dock_quality

    # ---- 完成事件奖励（一次性主导信号） ----
    # 完成条件：完全进入泊位 + 朝向误差 < 30° + 速度 < 0.05 m/s
    inside_dock = 1.0 if (abs(dx) <= tol_x and abs(dy) <= tol_y) else 0.0
    aligned = 1.0 if angle_err <= 0.5 else 0.0
    still = 1.0 if crate_speed < 0.05 else 0.0
    completed = inside_dock * aligned * still
    completion_bonus = 600.0 * completed

    # ---- 约束惩罚 ----
    # 接近泊位时抑制货箱速度（门控：仅当货箱接近泊位时激活）
    near_dock = 1.0 if (abs(dx) < 3.0 * tol_x and abs(dy) < 3.0 * tol_y) else 0.0
    speed_penalty_near_dock = -1.5 * near_dock * crate_speed

    # 越界惩罚（小车位置归一化坐标超出 [-1, 1] 边界）
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    oob = 0.0
    if abs(cart_x) > 1.0:
        oob += (abs(cart_x) - 1.0) ** 2
    if abs(cart_y) > 1.0:
        oob += (abs(cart_y) - 1.0) ** 2
    out_of_bounds_penalty = -20.0 * oob

    # 障碍接近惩罚（仅在接触附近时生效，避免压制正常推动）
    front = next_obs[15]
    left = next_obs[16]
    right = next_obs[17]
    obstacle_penalty = -0.5 * (max(0.0, front - 0.8) + max(0.0, left - 0.8) + max(0.0, right - 0.8))

    # 动作平滑（轻量，避免压制必要推动）
    action_smoothness = -0.02 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_progress": crate_progress,
        "crate_docking_quality": crate_docking_quality,
        "completion_bonus": completion_bonus,
        "speed_penalty_near_dock": speed_penalty_near_dock,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "obstacle_penalty": obstacle_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_progress
        + crate_docking_quality
        + completion_bonus
        + speed_penalty_near_dock
        + out_of_bounds_penalty
        + obstacle_penalty
        + action_smoothness
    )

    return float(total_reward), components