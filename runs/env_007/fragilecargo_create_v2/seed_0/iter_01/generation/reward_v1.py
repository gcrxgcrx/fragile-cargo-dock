def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到坞的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱到坞距离（归一化尺度）
    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向误差（弧度），坞假设与坐标轴对齐
    crate_heading = (obs[11] / (1.0 + 1e-6)) if False else 0.0  # placeholder, replaced below
    # 用 atan2 计算朝向角，再取与 0 的夹角（坞对齐假设为 0 朝向）
    import_guard = 0.0  # no-op, keep function pure
    ch = obs[10]
    sh = obs[11]
    # 朝向角
    heading_angle = 0.0
    # 手动 atan2 近似：使用 cos/sin 直接构造对齐度，避免 atan2 依赖
    # 对齐度 = cos(heading)（坞朝向假设为 0，即 cos=1 表示对齐）
    align_cos = ch  # 范围 [-1, 1]，1 表示完全对齐

    # 接触标志
    contact = obs[14]

    # 小车前向速度（归一化）
    cart_v = obs[4]

    # ---------- 组件 A: 货箱向坞的进度（delta 形式，防悬停） ----------
    # 用 improvement_delta：距离减少为正
    progress = (dist - next_dist)  # 正=靠近
    # 低速门控：接近坞时抑制速度，防止高速滑过
    # 门控因子在 dist 小时衰减，但不阻断早期探索（dist 大时接近 1）
    # 使用线性衰减门：dist < 0.15 时开始衰减
    near_gate = 1.0
    if dist < 0.15:
        near_gate = max(0.2, dist / 0.15)  # 最低保留 0.2，避免完全阻断
    # 货箱速率门控：接近坞时若速度高，抑制进度奖励
    speed_gate = 1.0
    if dist < 0.15:
        # 速度越高，门控越低（但保留最低 0.2）
        speed_gate = max(0.2, 1.0 / (1.0 + 2.0 * crate_speed))
    progress_reward = 5.0 * progress * near_gate * speed_gate

    # ---------- 组件 B: 货箱朝向对齐（仅在接近坞时启用） ----------
    # 对齐度：align_cos 在 [-1,1]，映射到 [0,1]
    align_factor = (align_cos + 1.0) * 0.5  # 0=反向, 1=对齐
    # 仅在货箱接近坞时启用（dist < 0.3）
    align_gate = 0.0
    if dist < 0.3:
        align_gate = max(0.0, 1.0 - dist / 0.3)
    alignment_reward = 0.5 * align_factor * align_gate

    # ---------- 组件 C: 货箱在坞内近静止（settling，仅在近坞时启用） ----------
    # 速率惩罚：仅在货箱接近坞时启用，避免抑制必要推动
    settle_gate = 0.0
    if dist < 0.2:
        settle_gate = max(0.0, 1.0 - dist / 0.2)
    # 速率超过 0.05 m/s 时惩罚（hinge 形式）
    speed_excess = max(0.0, crate_speed - 0.05)
    settling_penalty = -1.0 * speed_excess * settle_gate

    # ---------- 组件 D: 边界/障碍安全（hinge 形式，轻量） ----------
    # 小车位置越界风险：obs[0], obs[1] 接近 ±1 时惩罚
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    # 小车边界 hinge：|x| > 0.85 或 |y| > 0.85 时惩罚
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    # 障碍接近度惩罚：sensor 接近 1 时轻罚
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.8:
        boundary_penalty -= 0.3 * (sensor_max - 0.8)

    # ---------- 组件 E: 接触冲击抑制（保守，仅在接触时） ----------
    # 接触时若货箱速度高，轻罚（间接推断硬碰撞风险）
    impact_penalty = 0.0
    if contact > 0.5:
        # 接触时货箱速率超过 1.0 m/s 时轻罚
        impact_excess = max(0.0, crate_speed - 1.0)
        impact_penalty = -0.3 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + alignment_reward
        + settling_penalty
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_dock_alignment": float(alignment_reward),
        "crate_settling": float(settling_penalty),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components