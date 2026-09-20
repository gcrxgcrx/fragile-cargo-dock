def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主信号：货箱到泊位的进度（用 delta 距离，避免悬停陷阱）----
    d_cur = (obs[12] ** 2 + obs[13] ** 2) ** 0.5
    d_next = (next_obs[12] ** 2 + next_obs[13] ** 2) ** 0.5
    progress = (d_cur - d_next) * 8.0

    # ---- 泊位质量：位置接近度（平滑压缩，仅在接近时贡献）----
    near = 1.0 / (1.0 + 12.0 * d_next)
    inside_x = max(0.0, 1.0 - abs(next_obs[12]) / 0.05)
    inside_y = max(0.0, 1.0 - abs(next_obs[13]) / 0.06)
    inside = inside_x * inside_y

    # ---- 朝向对齐：货箱朝向误差（<30° 目标）----
    ang = (next_obs[11] ** 2 + next_obs[10] ** 2) ** 0.5
    if ang > 1e-6:
        cos_h = next_obs[10] / ang
        sin_h = next_obs[11] / ang
    else:
        cos_h = 1.0
        sin_h = 0.0
    align = max(0.0, (cos_h - 0.5) / 0.5)  # cos<0.5 时为 0，cos=1 时为 1

    # ---- 静止：货箱速度（仅在接近泊位时启用）----
    spd = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    slow = 1.0 / (1.0 + 8.0 * spd)

    # 联合完成信号（连续 proxy）：进入泊位 * 对齐 * 静止
    dock_quality = inside * align * slow

    # ---- 软接触：仅在接触且货箱速度突变较大时轻罚（避免误伤正常推动）----
    contact = next_obs[14]
    spd_prev = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5
    spd_jump = abs(spd - spd_prev)
    soft_contact = -0.05 * contact * max(0.0, spd_jump - 0.5)

    # ---- 越界防护：小车与货箱位置的 hinge 惩罚 ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_edge = max(0.0, abs(cart_x) - 0.9) + max(0.0, abs(cart_y) - 0.9)
    # 货箱世界坐标恢复
    cch = next_obs[2]
    csh = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_wx = rel_x * cch - rel_y * csh + cart_x * 5.0
    crate_wy = rel_x * csh + rel_y * cch + cart_y * 4.0
    crate_edge = max(0.0, abs(crate_wx) / 5.0 - 0.9) + max(0.0, abs(crate_wy) / 4.0 - 0.9)
    out_of_bounds = -0.3 * (cart_edge + crate_edge)

    # ---- 障碍接近度惩罚（前方静态障碍）----
    obstacle = -0.05 * max(0.0, next_obs[15] - 0.7)

    # ---- 动作平滑（轻量）----
    smooth = -0.01 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_to_dock_progress": progress,
        "crate_docking_quality": 0.5 * dock_quality,
        "crate_near_dock": 0.2 * near,
        "soft_contact": soft_contact,
        "out_of_bounds": out_of_bounds,
        "obstacle_penalty": obstacle,
        "action_smoothness": smooth,
    }
    total = sum(components.values())
    return float(total), components