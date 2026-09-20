def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 读取观测（仅使用已声明维度）----
    cart_x = obs[0]
    cart_y = obs[1]

    crate_vx = obs[8]
    crate_vy = obs[9]
    crate_cos = obs[10]
    crate_sin = obs[11]
    crate_to_dock_x = obs[12]
    crate_to_dock_y = obs[13]
    contact = obs[14]

    n_crate_vx = next_obs[8]
    n_crate_vy = next_obs[9]
    n_crate_cos = next_obs[10]
    n_crate_sin = next_obs[11]
    n_crate_to_dock_x = next_obs[12]
    n_crate_to_dock_y = next_obs[13]

    drive = action[0]
    steer = action[1]

    # ---- 货箱到泊位的归一化距离 ----
    dist = (crate_to_dock_x ** 2 + crate_to_dock_y ** 2) ** 0.5
    n_dist = (n_crate_to_dock_x ** 2 + n_crate_to_dock_y ** 2) ** 0.5

    # ---- 货箱速度（归一化单位）----
    crate_speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5
    n_crate_speed = (n_crate_vx ** 2 + n_crate_vy ** 2) ** 0.5

    # ---- 货箱朝向误差（弧度，范围 0~pi）----
    # 泊位期望朝向为 +x 方向（cos=1, sin=0）
    heading_err = (crate_sin ** 2 + (1.0 - crate_cos) ** 2) ** 0.5
    n_heading_err = (n_crate_sin ** 2 + (1.0 - n_crate_cos) ** 2) ** 0.5

    # =========================================================
    # 1) crate_to_dock_progress : 货箱向泊位靠近的进度信号
    #    使用 improvement_delta（避免悬停陷阱）
    # =========================================================
    progress = (dist - n_dist) * 10.0
    progress = max(-2.0, min(2.0, progress))

    # =========================================================
    # 2) crate_docking_quality : 泊位附近的位置/朝向/静止联合质量
    #    分项 bounded factor，用几何平均避免塌缩
    # =========================================================
    # 位置因子：距离越小越接近 1
    pos_factor = 1.0 / (1.0 + 8.0 * n_dist)

    # 朝向因子：朝向误差越小越接近 1（30deg ~ 0.52 rad 附近明显衰减）
    head_factor = 1.0 / (1.0 + 6.0 * n_heading_err)

    # 静止因子：速度越小越接近 1
    speed_factor = 1.0 / (1.0 + 12.0 * n_crate_speed)

    # 靠近泊位时才强调静止与对齐（用位置因子做门控，连续）
    near_gate = 1.0 / (1.0 + 4.0 * n_dist)

    docking_quality = (
        (pos_factor + head_factor + speed_factor) / 3.0
    ) ** 1.0

    # 靠近泊位时额外的对齐 + 静止 shaping
    near_align = near_gate * head_factor
    near_still = near_gate * speed_factor

    docking_reward = 0.6 * docking_quality + 0.8 * near_align + 0.8 * near_still

    # =========================================================
    # 3) crate_speed_penalty_near_dock : 接近泊位时抑制货箱速度
    #    仅在接近时启用（hinge 风格门控）
    # =========================================================
    speed_penalty = -1.2 * near_gate * (n_crate_speed ** 2)

    # =========================================================
    # 4) soft_contact_penalty : 接触时的货箱速度突变（间接推断硬碰撞）
    #    仅当发生接触时生效
    # =========================================================
    speed_jump = abs(n_crate_speed - crate_speed)
    soft_contact_penalty = 0.0
    if contact > 0.5:
        soft_contact_penalty = -0.5 * speed_jump

    # =========================================================
    # 5) out_of_bounds_penalty : 小车 / 货箱接近场地边界
    #    使用 hinge 形式，仅在接近边界时惩罚
    # =========================================================
    # 小车位置归一化约在 [-1, 1]（0=中线），边界设为 0.85
    cart_edge = max(abs(cart_x), abs(cart_y))
    cart_bound_penalty = -1.0 * max(0.0, cart_edge - 0.85)

    # 货箱位置（世界系）由相对位置 + 小车位置恢复
    # 车体坐标 -> 世界系旋转
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    world_crate_x = cart_x * 5.0 + (rel_x * obs[2] - rel_y * obs[3])
    world_crate_y = cart_y * 4.0 + (rel_x * obs[3] + rel_y * obs[2])
    crate_edge = max(abs(world_crate_x) / 5.0, abs(world_crate_y) / 4.0)
    crate_bound_penalty = -1.0 * max(0.0, crate_edge - 0.9)

    # =========================================================
    # 6) action_smoothness : 动作幅度轻量约束（可选，权重小）
    # =========================================================
    action_penalty = -0.02 * (drive ** 2 + steer ** 2)

    # ---- 汇总 ----
    components = {
        "crate_to_dock_progress": float(progress),
        "crate_docking_quality": float(docking_reward),
        "crate_speed_penalty_near_dock": float(speed_penalty),
        "soft_contact_penalty": float(soft_contact_penalty),
        "cart_out_of_bounds_penalty": float(cart_bound_penalty),
        "crate_out_of_bounds_penalty": float(crate_bound_penalty),
        "action_smoothness": float(action_penalty),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)