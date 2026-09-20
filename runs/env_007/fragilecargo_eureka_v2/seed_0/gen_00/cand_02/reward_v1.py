def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 观测解包（仅使用声明索引） ----------
    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0
    next_crate_vx = next_obs[8] * 3.0
    next_crate_vy = next_obs[9] * 3.0

    # 货箱到 dock 中心偏移（有符号，单位 m）
    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    next_dx = next_obs[12] * 5.0
    next_dy = next_obs[13] * 4.0

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (next_dx * next_dx + next_dy * next_dy) ** 0.5

    # 货箱朝向（世界系）
    crate_heading = 0.0
    if obs[10] != 0.0 or obs[11] != 0.0:
        crate_heading = obs[11]
        crate_heading = crate_heading  # placeholder, 使用 cos/sin 组合
    cos_h = obs[10]
    sin_h = obs[11]
    # 假设 dock 朝向为世界 x 轴（0 弧度），对齐度 = cos(heading)
    align_cos = cos_h
    # 归一化到 [0,1]
    align = 0.5 * (align_cos + 1.0)

    next_cos_h = next_obs[10]
    next_align = 0.5 * (next_cos_h + 1.0)

    # 货箱速度模长
    speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 接触与小车速度
    contact = obs[14]
    cart_speed = obs[4] * 3.0
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]

    # ---------- 组件 1：货箱向 dock 的进度（delta 距离） ----------
    progress = (dist - next_dist) * 3.0
    # 限制单步幅度，避免噪声
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5

    # ---------- 组件 2：朝向对齐进度（delta） ----------
    align_delta = (next_align - align) * 0.5

    # ---------- 组件 3：停靠联合条件 proxy（近 + 对齐 + 静止） ----------
    # 距离因子：越近越大
    near_factor = 1.0 / (1.0 + 2.0 * next_dist)
    # 对齐因子
    align_factor = next_align
    # 静止因子：速度低于 0.05 时接近 1
    next_speed = (next_crate_vx * next_crate_vx + next_crate_vy * next_crate_vy) ** 0.5
    still_factor = 1.0 / (1.0 + 10.0 * next_speed)
    # 几何平均，避免塌缩
    dock_proxy = (near_factor * align_factor * still_factor) ** (1.0 / 3.0)
    # 只在货箱接近 dock 时才给停靠奖励（门控）
    if next_dist < 1.5:
        dock_reward = 1.5 * dock_proxy
    else:
        dock_reward = 0.0

    # ---------- 组件 4：轻柔接触（门控惩罚，仅接触且相对速度高时） ----------
    # 相对速度近似：小车速度与货箱速度差
    rel_speed = 0.0
    if contact > 0.5:
        rel_speed = (cart_speed * cart_speed + next_speed * next_speed) ** 0.5
    # 仅在接触且相对速度超过阈值时惩罚（hinge）
    gentle_penalty = 0.0
    if contact > 0.5:
        excess = rel_speed - 1.5
        if excess > 0.0:
            gentle_penalty = -0.3 * excess

    # ---------- 组件 5：边界规避（hinge，仅接近边界时） ----------
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    # 小车归一化位置，越接近 ±1 越危险
    if cart_x > 0.75:
        boundary_penalty -= 0.2 * (cart_x - 0.75) / 0.25
    if cart_x < -0.75:
        boundary_penalty -= 0.2 * (-0.75 - cart_x) / 0.25
    if cart_y > 0.75:
        boundary_penalty -= 0.2 * (cart_y - 0.75) / 0.25
    if cart_y < -0.75:
        boundary_penalty -= 0.2 * (-0.75 - cart_y) / 0.25
    # 货箱边界（用 dock 偏移间接判断，货箱偏移大说明远）
    if next_dist > 4.0:
        boundary_penalty -= 0.1 * (next_dist - 4.0)

    # ---------- 组件 6：静态障碍接近惩罚（hinge） ----------
    obstacle_penalty = 0.0
    if sensor_front > 0.85:
        obstacle_penalty -= 0.15 * (sensor_front - 0.85) / 0.15
    if sensor_left > 0.9:
        obstacle_penalty -= 0.1 * (sensor_left - 0.9) / 0.1
    if sensor_right > 0.9:
        obstacle_penalty -= 0.1 * (sensor_right - 0.9) / 0.1

    # ---------- 组件 7：动作平滑（轻量，避免抖动） ----------
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {}
    components["crate_to_dock_progress"] = progress
    components["crate_dock_alignment"] = align_delta
    components["dock_proxy_reward"] = dock_reward
    components["gentle_contact_penalty"] = gentle_penalty
    components["boundary_penalty"] = boundary_penalty
    components["obstacle_penalty"] = obstacle_penalty
    components["action_smoothness_penalty"] = smooth_penalty

    total_reward = (
        components["crate_to_dock_progress"]
        + components["crate_dock_alignment"]
        + components["dock_proxy_reward"]
        + components["gentle_contact_penalty"]
        + components["boundary_penalty"]
        + components["obstacle_penalty"]
        + components["action_smoothness_penalty"]
    )

    return float(total_reward), components