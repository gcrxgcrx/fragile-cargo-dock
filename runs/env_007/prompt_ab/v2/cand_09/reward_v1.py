def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 几何常量（来自环境事实） ----------
    HALF_W = 5.0          # 仓库半宽 (m)
    HALF_H = 4.0          # 仓库半高 (m)
    DOCK_TOL_X = 0.024    # |obs[12]| <= 0.024 视为 x 方向完全进入
    DOCK_TOL_Y = 0.030    # |obs[13]| <= 0.030 视为 y 方向完全进入
    DOCK_RADIUS = 0.15    # 接近泊位的门控半径（归一化单位，远小于 1，仅用于 shaping 门控）

    # ---------- 货箱到泊位的归一化偏移 ----------
    dx = next_obs[12]
    dy = next_obs[13]
    dx_prev = obs[12]
    dy_prev = obs[13]

    # 归一化距离（用半宽/半高恢复米制后求距离，再除以半宽归一化）
    mx = dx * HALF_W
    my = dy * HALF_H
    mx_prev = dx_prev * HALF_W
    my_prev = dy_prev * HALF_H
    dist = (mx * mx + my * my) ** 0.5
    dist_prev = (mx_prev * mx_prev + my_prev * my_prev) ** 0.5

    # ---------- 1) 主进度：货箱向泊位靠近的增量信号 ----------
    # 只在“这一帧更接近了”时给分，避免悬停在泊位附近持续收割
    progress_raw = dist_prev - dist
    if progress_raw > 0.0:
        crate_progress = 3.0 * progress_raw
    else:
        crate_progress = 0.0

    # ---------- 2) 泊位停靠质量（联合条件软代理） ----------
    # 位置因子：完全进入容差区才接近 1
    pos_err_x = abs(dx) / DOCK_TOL_X
    pos_err_y = abs(dy) / DOCK_TOL_Y
    pos_factor = 1.0 / (1.0 + pos_err_x * pos_err_x + pos_err_y * pos_err_y)

    # 朝向因子：货箱朝向误差 < 30° 才接近 1
    crate_heading = (next_obs[11] / (1.0 if abs(next_obs[11]) > 0.0 else 1.0))
    # 使用 atan2 语义：cos, sin 已归一化，直接计算角度误差
    cos_h = next_obs[10]
    sin_h = next_obs[11]
    # 归一化朝向向量（防止非单位向量）
    norm_h = (cos_h * cos_h + sin_h * sin_h) ** 0.5
    if norm_h > 1e-6:
        cos_h = cos_h / norm_h
        sin_h = sin_h / norm_h
    # 泊位目标朝向为 0 弧度（对齐 x 轴），误差角 = atan2(sin, cos)
    # 用 cos(误差) 做连续因子：误差 0° -> 1，误差 30° -> cos(30°)=0.866，误差 90° -> 0
    heading_factor = max(0.0, cos_h)  # 误差 < 90° 时为正，30° 时约 0.866
    # 更精确：cos(30°) = 0.8660254，用它做归一化使 30° 处因子≈1
    heading_factor = heading_factor / 0.8660254
    if heading_factor > 1.0:
        heading_factor = 1.0

    # 速度因子：货箱速度 < 0.05 m/s 才接近 1
    # 货箱速度归一化值 obs[8], obs[9] 是 /3.0 的，恢复米制速度
    vx = next_obs[8] * 3.0
    vy = next_obs[9] * 3.0
    speed = (vx * vx + vy * vy) ** 0.5
    # 速度阈值 0.05 m/s，但 shaping 用平滑衰减：speed=0 -> 1，speed=0.5 -> 接近 0
    speed_factor = 1.0 / (1.0 + (speed / 0.05) * (speed / 0.05))

    # 联合因子（几何平均，避免乘积塌缩）
    dock_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    # 仅在货箱接近泊位时激活（门控），避免远处就收分
    near_gate = 1.0 if dist < DOCK_RADIUS else 0.0
    crate_docking_quality = 2.0 * dock_quality * near_gate

    # ---------- 3) 完成事件奖励（一次性大额） ----------
    # 完成条件：完全进入 + 朝向误差 < 30° + 速度 < 0.05 m/s
    # 从 obs 显式推断，容差严格取自环境事实
    inside_x = abs(dx) <= DOCK_TOL_X
    inside_y = abs(dy) <= DOCK_TOL_Y
    heading_ok = cos_h >= 0.8660254   # 误差 < 30°
    speed_ok = speed < 0.05
    # 用连续乘积形式（但只在真正满足时给大额，用二值门控确保一次性）
    if inside_x and inside_y and heading_ok and speed_ok:
        completion_bonus = 5000.0
    else:
        completion_bonus = 0.0

    # ---------- 4) 软接触惩罚（间接推断，轻量） ----------
    # 接触标志 + 货箱速度突变：接触时货箱速度过大视为硬碰撞风险
    contact = next_obs[14]
    if contact > 0.5:
        # 接触时货箱速度越大，越可能是硬碰撞
        soft_contact_penalty = -0.5 * (speed / 3.0) ** 2
    else:
        soft_contact_penalty = 0.0

    # ---------- 5) 泊位附近速度抑制（仅在接近泊位时启用） ----------
    # 避免货箱滑过泊位；只在接近泊位时惩罚高速
    if dist < DOCK_RADIUS:
        crate_speed_penalty = -1.0 * (speed / 3.0) ** 2
    else:
        crate_speed_penalty = 0.0

    # ---------- 6) 越界惩罚（hinge 形式，仅在接近边界时生效） ----------
    # 小车位置 obs[0], obs[1] 是归一化到 [-2,2] 的，边界在 |x|>1 或 |y|>1 附近
    cart_x = obs[0]
    cart_y = obs[1]
    # 货箱世界坐标恢复：crate_rel_body -> 世界系
    cart_cos = obs[2]
    cart_sin = obs[3]
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    crate_wx = cart_cos * rel_x - cart_sin * rel_y + cart_x * HALF_W
    crate_wy = cart_sin * rel_x + cart_cos * rel_y + cart_y * HALF_H
    crate_nx = crate_wx / HALF_W
    crate_ny = crate_wy / HALF_H

    # 边界余量：|归一化坐标| > 0.85 时开始惩罚
    bound_margin = 0.85
    out_penalty = 0.0
    for coord in (cart_x, cart_y, crate_nx, crate_ny):
        excess = abs(coord) - bound_margin
        if excess > 0.0:
            out_penalty -= 0.5 * excess * excess

    # ---------- 7) 动作平滑（轻量，可选） ----------
    action_smoothness = -0.01 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {
        "crate_progress": crate_progress,
        "crate_docking_quality": crate_docking_quality,
        "completion_bonus": completion_bonus,
        "soft_contact_penalty": soft_contact_penalty,
        "crate_speed_penalty": crate_speed_penalty,
        "out_of_bounds_penalty": out_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_progress
        + crate_docking_quality
        + completion_bonus
        + soft_contact_penalty
        + crate_speed_penalty
        + out_penalty
        + action_smoothness
    )

    return float(total_reward), components