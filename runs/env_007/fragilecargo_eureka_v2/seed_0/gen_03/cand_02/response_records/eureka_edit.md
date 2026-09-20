# Response Record

分析：任务分数 8.09 但 20/20 全部 truncation，说明策略从未真正完成停靠——它学会了在 dock 附近"悬停收割"`dock_proxy_reward`（episode 均值 707，占总奖励 90%+），而非满足"进入 dock + 对齐 + 静止保持"的终止条件。`dock_proxy_reward` 的乘积形式在 `next_dist<0.8` 时给 6.0 分且随接近陡增，即使货箱仍在动、朝向未对齐也能拿高分，是典型悬停陷阱。`crate_to_dock_progress` 量级太小（12 vs 707）几乎无梯度。改法：把主信号改为 delta 距离并放大权重，把 dock_proxy 改为"近+对齐+静止"的严格几何平均且要求货箱真正低速才给分（低速作为门控而非持续奖励），并用时间进度做轻微衰减防止拖延。

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 观测解包 ----------
    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0
    next_crate_vx = next_obs[8] * 3.0
    next_crate_vy = next_obs[9] * 3.0

    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    next_dx = next_obs[12] * 5.0
    next_dy = next_obs[13] * 4.0

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (next_dx * next_dx + next_dy * next_dy) ** 0.5

    # 货箱朝向对齐（假设 dock 朝向为世界 x 轴）
    cos_h = obs[10]
    next_cos_h = next_obs[10]
    align = 0.5 * (cos_h + 1.0)
    next_align = 0.5 * (next_cos_h + 1.0)

    speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    next_speed = (next_crate_vx * next_crate_vx + next_crate_vy * next_crate_vy) ** 0.5

    contact = obs[14]
    cart_speed = obs[4] * 3.0
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    time_fraction = obs[18]

    # ---------- 组件 1：货箱向 dock 的进度（主信号，delta 距离放大） ----------
    raw_progress = dist - next_dist
    # 正向进度放大，负向进度保留但略轻，鼓励持续靠近
    if raw_progress > 0.0:
        progress = 8.0 * raw_progress
    else:
        progress = 4.0 * raw_progress
    if progress > 1.5:
        progress = 1.5
    if progress < -1.5:
        progress = -1.5

    # ---------- 组件 2：朝向对齐（门控在接近 dock 时，作为引导） ----------
    align_reward = 0.0
    if next_dist < 2.0:
        align_reward = 0.5 * next_align

    # ---------- 组件 3：停靠联合条件（近 + 对齐 + 静止），严格门槛 ----------
    # 每个 factor 连续 bounded；静止 factor 用 hinge 式，只有真正低速才接近 1
    near_factor = 1.0 / (1.0 + 6.0 * next_dist)
    align_factor = next_align
    # 货箱速度 < 0.05 视为静止；用衰减使低速才得分
    still_factor = 1.0 / (1.0 + 40.0 * next_speed)
    dock_proxy = (near_factor * align_factor * still_factor) ** (1.0 / 3.0)
    # 仅在货箱真正接近 dock 中心时给奖励，且要求货箱低速（门控）
    dock_reward = 0.0
    if next_dist < 0.6 and next_speed < 0.5:
        dock_reward = 8.0 * dock_proxy

    # ---------- 组件 4：轻柔接触（门控 hinge 惩罚） ----------
    gentle_penalty = 0.0
    if contact > 0.5:
        rel_speed = (cart_speed * cart_speed + next_speed * next_speed) ** 0.5
        excess = rel_speed - 1.5
        if excess > 0.0:
            gentle_penalty = -0.3 * excess

    # ---------- 组件 5：边界规避（hinge） ----------
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    if cart_x > 0.75:
        boundary_penalty -= 0.2 * (cart_x - 0.75) / 0.25
    if cart_x < -0.75:
        boundary_penalty -= 0.2 * (-0.75 - cart_x) / 0.25
    if cart_y > 0.75:
        boundary_penalty -= 0.2 * (cart_y - 0.75) / 0.25
    if cart_y < -0.75:
        boundary_penalty -= 0.2 * (-0.75 - cart_y) / 0.25
    if next_dist > 4.0:
        boundary_penalty -= 0.1 * (next_dist - 4.0)

    # ---------- 组件 6：静态障碍接近惩罚（hinge） ----------
    obstacle_penalty = 0.0
    if sensor_front > 0.6:
        obstacle_penalty -= 0.2 * (sensor_front - 0.6) / 0.4
    if sensor_left > 0.7:
        obstacle_penalty -= 0.15 * (sensor_left - 0.7) / 0.3
    if sensor_right > 0.7:
        obstacle_penalty -= 0.15 * (sensor_right - 0.7) / 0.3

    # ---------- 组件 7：动作平滑（轻量） ----------
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 组件 8：时间效率（轻微，防止拖延到 truncation） ----------
    time_penalty = -0.05 * time_fraction

    # ---------- 汇总 ----------
    components = {}
    components["crate_to_dock_progress"] = progress
    components["crate_dock_alignment"] = align_reward
    components["dock_proxy_reward"] = dock_reward
    components["gentle_contact_penalty"] = gentle_penalty
    components["boundary_penalty"] = boundary_penalty
    components["obstacle_penalty"] = obstacle_penalty
    components["action_smoothness_penalty"] = smooth_penalty
    components["time_penalty"] = time_penalty

    total_reward = (
        components["crate_to_dock_progress"]
        + components["crate_dock_alignment"]
        + components["dock_proxy_reward"]
        + components["gentle_contact_penalty"]
        + components["boundary_penalty"]
        + components["obstacle_penalty"]
        + components["action_smoothness_penalty"]
        + components["time_penalty"]
    )

    return float(total_reward), components
