def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 恢复关键量 ----
    # 货箱到泊位的有符号偏移（归一化）
    dx = next_obs[12]
    dy = next_obs[13]
    dx_prev = obs[12]
    dy_prev = obs[13]

    # 泊位偏移距离（归一化欧氏距离）
    dist = (dx * dx + dy * dy) ** 0.5
    dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5

    # 货箱世界速度（归一化）
    vx = next_obs[8]
    vy = next_obs[9]
    speed = (vx * vx + vy * vy) ** 0.5

    # 货箱朝向误差（弧度）
    crate_heading = 0.0
    if next_obs[10] != 0.0 or next_obs[11] != 0.0:
        crate_heading = (next_obs[11] ** 2 + next_obs[10] ** 2) ** 0.5
    # 朝向误差用 cos 分量：目标朝向为 0（cos=1, sin=0）
    cos_err = 1.0 - next_obs[10]
    # 归一化朝向误差 [0,1]
    heading_err = cos_err * 0.5

    contact = next_obs[14]

    # ---- 1. 货箱到泊位进度（主信号，delta 形式，避免悬停陷阱）----
    progress = dist_prev - dist
    crate_to_dock_progress = 6.0 * progress

    # ---- 2. 泊位接近度 shaping（门控在接近时激活）----
    # 用 bounded 衰减，距离越近越大，仅作为引导
    proximity = 1.0 / (1.0 + 10.0 * dist)
    crate_dock_proximity = 1.0 * proximity

    # ---- 3. 停靠质量：接近泊位时激活的联合条件 ----
    # 位置因子：越接近泊位中心越好
    pos_factor = 1.0 / (1.0 + 20.0 * dist)
    # 朝向因子：朝向误差越小越好
    heading_factor = 1.0 / (1.0 + 3.0 * heading_err)
    # 速度因子：速度越小越好
    speed_factor = 1.0 / (1.0 + 8.0 * speed)
    # 几何平均，避免塌缩
    docking_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    # 仅在接近泊位时激活（门控）
    near_gate = 1.0 / (1.0 + 6.0 * dist)
    crate_docking_quality = 2.0 * docking_quality * near_gate

    # ---- 4. 接近泊位时的速度抑制（门控：仅在接近时激活）----
    # 防止货箱滑过泊位
    speed_penalty = -1.5 * speed * near_gate

    # ---- 5. 软接触惩罚（间接推断，轻量）----
    # 接触且货箱速度大时，可能为硬碰撞
    soft_contact_penalty = 0.0
    if contact > 0.5:
        soft_contact_penalty = -0.3 * speed

    # ---- 6. 越界惩罚（hinge，仅接近边界时激活）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_edge = max(0.0, abs(cart_x) - 0.85) + max(0.0, abs(cart_y) - 0.85)
    # 货箱越界：用泊位偏移无法直接判断，使用相对位置恢复
    crate_rel_x = next_obs[6] * 3.0
    crate_rel_y = next_obs[7] * 3.0
    # 小车位置（米）
    cart_x_m = cart_x * 5.0
    cart_y_m = cart_y * 4.0
    # 货箱世界位置（米）：车体系旋转到世界系
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    crate_wx = cart_x_m + cos_h * crate_rel_x - sin_h * crate_rel_y
    crate_wy = cart_y_m + sin_h * crate_rel_x + cos_h * crate_rel_y
    # 货箱越界（仓库半宽5、半高4）
    crate_edge = max(0.0, abs(crate_wx) / 5.0 - 0.85) + max(0.0, abs(crate_wy) / 4.0 - 0.85)
    out_of_bounds_penalty = -1.0 * (cart_edge + crate_edge)

    # ---- 7. 动作平滑（轻量）----
    action_smoothness = -0.05 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_dock_proximity": float(crate_dock_proximity),
        "crate_docking_quality": float(crate_docking_quality),
        "speed_penalty_near_dock": float(speed_penalty),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components