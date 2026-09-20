def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到泊位的有符号偏移（归一化）
    cx = next_obs[12]
    cy = next_obs[13]
    px = obs[12]
    py = obs[13]

    # 货箱世界系速度（归一化后）
    vx = next_obs[8]
    vy = next_obs[9]
    crate_speed = (vx * vx + vy * vy) ** 0.5

    # 货箱朝向误差（弧度）
    import_angle_error = 0.0  # 占位，避免 import；实际用 atan2 替代
    # 用 cos/sin 计算朝向误差的余弦（无需 atan2）
    ch = next_obs[10]
    sh = next_obs[11]
    # 泊位朝向假定为 +x 方向（cos=1, sin=0），误差 cos = ch
    heading_align = ch  # in [-1, 1]，1 表示完全对齐

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    sf = next_obs[15]
    sl = next_obs[16]
    sr = next_obs[17]

    # 小车位置
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # 动作
    a0 = action[0]
    a1 = action[1]

    # ---------- 距离度量 ----------
    prev_dist = (px * px + py * py) ** 0.5
    curr_dist = (cx * cx + cy * cy) ** 0.5

    # 进入泊位的连续因子（|cx|<=0.024, |cy|<=0.030 为完全进入）
    in_x = max(0.0, 1.0 - abs(cx) / 0.024)
    in_y = max(0.0, 1.0 - abs(cy) / 0.030)
    inside_factor = min(in_x, in_y)  # 0~1，完全进入=1

    # 朝向对齐因子（误差 < 30° 即 cos > 0.866）
    align_factor = max(0.0, (heading_align - 0.5) / 0.5)  # 0~1

    # 静止因子（速度 < 0.05 m/s 对应归一化 0.05/3.0 ≈ 0.0167）
    speed_thresh = 0.0167
    slow_factor = max(0.0, 1.0 - crate_speed / (speed_thresh * 3.0))

    # 接近泊位因子（用于门控速度惩罚）
    near_dock = max(0.0, 1.0 - curr_dist / 0.15)

    # ---------- 组件 1：货箱向泊位推进（主信号，delta 形式） ----------
    progress = (prev_dist - curr_dist) * 5.0
    # 限制单步幅度，避免震荡刷分
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5

    # ---------- 组件 2：进入泊位的联合条件代理 ----------
    # 几何平均避免塌缩
    dock_quality = 0.0
    if inside_factor > 0.0 or align_factor > 0.0 or slow_factor > 0.0:
        prod = (inside_factor + 1e-6) * (align_factor + 1e-6) * (slow_factor + 1e-6)
        dock_quality = prod ** (1.0 / 3.0)
    dock_quality = dock_quality * 2.0

    # ---------- 组件 3：接近泊位时的速度抑制（门控） ----------
    # 仅在接近泊位时惩罚高速，避免阻碍推进
    speed_penalty = -1.0 * near_dock * (crate_speed ** 2)

    # ---------- 组件 4：朝向对齐 shaping（接近泊位时激活） ----------
    align_shaping = 0.5 * near_dock * align_factor

    # ---------- 组件 5：软接触惩罚（间接推断硬碰撞） ----------
    # 接触时货箱速度突变大 → 可能硬碰撞
    soft_penalty = 0.0
    if contact > 0.5:
        # 货箱速度与小车前向速度差异大时惩罚
        cart_fwd = next_obs[4] * 3.0
        crate_fwd = vx * 3.0
        mismatch = abs(cart_fwd - crate_fwd)
        if mismatch > 0.5:
            soft_penalty = -0.3 * min(1.0, (mismatch - 0.5) / 1.5)

    # ---------- 组件 6：越界惩罚（hinge） ----------
    bound_penalty = 0.0
    # 小车边界（归一化位置，接近 ±1 时惩罚）
    if abs(cart_x) > 0.85:
        bound_penalty -= 0.5 * (abs(cart_x) - 0.85) / 0.15
    if abs(cart_y) > 0.85:
        bound_penalty -= 0.5 * (abs(cart_y) - 0.85) / 0.15
    # 货箱到泊位偏移过大说明可能越界（泊位在仓库内）
    if curr_dist > 0.8:
        bound_penalty -= 0.3 * (curr_dist - 0.8) / 0.2

    # ---------- 组件 7：障碍接近惩罚（hinge） ----------
    obstacle_penalty = 0.0
    if sf > 0.7:
        obstacle_penalty -= 0.2 * (sf - 0.7) / 0.3
    if sl > 0.7:
        obstacle_penalty -= 0.1 * (sl - 0.7) / 0.3
    if sr > 0.7:
        obstacle_penalty -= 0.1 * (sr - 0.7) / 0.3

    # ---------- 组件 8：动作平滑（轻量） ----------
    smooth_penalty = -0.02 * (a0 * a0 + a1 * a1)

    # ---------- 汇总 ----------
    components = {}
    components["crate_progress"] = float(progress)
    components["dock_quality"] = float(dock_quality)
    components["speed_penalty_near_dock"] = float(speed_penalty)
    components["align_shaping"] = float(align_shaping)
    components["soft_contact_penalty"] = float(soft_penalty)
    components["out_of_bounds_penalty"] = float(bound_penalty)
    components["obstacle_penalty"] = float(obstacle_penalty)
    components["action_smoothness"] = float(smooth_penalty)

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)