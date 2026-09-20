def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 主信号：货箱到泊位的接近进度 ----------
    # obs[12]: 货箱到泊位 x 偏移 / 半宽, obs[13]: y 偏移 / 半高
    dist_now = (obs[12] ** 2 + obs[13] ** 2) ** 0.5
    dist_next = (next_obs[12] ** 2 + next_obs[13] ** 2) ** 0.5
    dock_progress = (dist_now - dist_next) * 10.0

    # ---------- 泊位内几何质量：位置 + 朝向 ----------
    # 位置精度：越接近泊位中心越大（连续 bounded）
    pos_precision = 1.0 / (1.0 + 20.0 * dist_next)

    # 朝向误差：货箱朝向 vs 期望朝向（泊位对齐，假设期望为 0 弧度）
    crate_heading_err = abs(next_obs[11])  # sin 分量近似小角度误差
    heading_align = 1.0 / (1.0 + 6.0 * crate_heading_err)

    # ---------- 泊位内速度抑制（仅在接近泊位时启用） ----------
    crate_speed = (next_obs[8] ** 2 + next_obs[9] ** 2) ** 0.5
    near_dock_factor = 1.0 / (1.0 + 15.0 * dist_next)  # 越近越接近 1
    speed_quiet = 1.0 / (1.0 + 30.0 * crate_speed)

    # ---------- 联合停靠质量 proxy（几何平均避免塌缩） ----------
    dock_quality = (pos_precision * heading_align * speed_quiet) ** (1.0 / 3.0)
    dock_quality_reward = dock_quality * 2.0

    # 仅在接近泊位时才给速度抑制信号
    near_dock_speed_penalty = -1.0 * near_dock_factor * crate_speed

    # ---------- 软接触惩罚（间接推断：接触 + 货箱速度突变） ----------
    contact_now = obs[14]
    contact_next = next_obs[14]
    # 接触时货箱速度较大 -> 可能是硬推
    contact_speed_penalty = -0.3 * contact_next * crate_speed

    # ---------- 越界防护：小车与货箱接近边界 ----------
    # 小车位置 obs[0], obs[1]（归一化到半宽/半高）
    cart_edge = max(abs(next_obs[0]), abs(next_obs[1]))
    cart_edge_penalty = -0.5 * max(0.0, cart_edge - 0.85)

    # 货箱位置恢复：由 obs[6],[7] 车体系 + 小车朝向旋转后加小车位置
    # 车体系 -> 世界系
    c = next_obs[2]
    s = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_world_x = next_obs[0] * 5.0 + (c * rel_x - s * rel_y)
    crate_world_y = next_obs[1] * 4.0 + (s * rel_x + c * rel_y)
    # 归一化到半宽/半高（半宽约 5.0，半高约 4.0）
    crate_nx = crate_world_x / 5.0
    crate_ny = crate_world_y / 4.0
    crate_edge = max(abs(crate_nx), abs(crate_ny))
    crate_edge_penalty = -0.5 * max(0.0, crate_edge - 0.85)

    # ---------- 动作平滑（轻量） ----------
    action_penalty = -0.02 * (action[0] ** 2 + action[1] ** 2)

    # ---------- 时间压力（轻量，仅在接近尾声时） ----------
    time_frac = next_obs[18]
    time_penalty = -0.2 * max(0.0, time_frac - 0.8)

    components = {
        "dock_progress": dock_progress,
        "dock_quality": dock_quality_reward,
        "near_dock_speed_penalty": near_dock_speed_penalty,
        "contact_speed_penalty": contact_speed_penalty,
        "cart_edge_penalty": cart_edge_penalty,
        "crate_edge_penalty": crate_edge_penalty,
        "action_penalty": action_penalty,
        "time_penalty": time_penalty,
    }

    total_reward = (
        dock_progress
        + dock_quality_reward
        + near_dock_speed_penalty
        + contact_speed_penalty
        + cart_edge_penalty
        + crate_edge_penalty
        + action_penalty
        + time_penalty
    )

    return (float(total_reward), components)