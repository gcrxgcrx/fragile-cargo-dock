def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 恢复关键量 ----
    # 货箱到泊位偏移（归一化 -> 米）
    dx_cur = obs[12] * 5.0
    dy_cur = obs[13] * 4.0
    dx_next = next_obs[12] * 5.0
    dy_next = next_obs[13] * 4.0

    dist_cur = (dx_cur * dx_cur + dy_cur * dy_cur) ** 0.5
    dist_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5

    # 货箱速度（世界系，归一化 -> m/s）
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 小车前向速度
    cart_fwd_speed = obs[4] * 3.0

    # 接触标志
    contact = obs[14]

    # ---- 组件 A: 货箱向泊位推进（增量信号，避免悬停刷分）----
    # 用 improvement_delta：只有"更接近"才得分，占据某状态本身不得分
    progress_delta = dist_cur - dist_next  # 正=靠近
    # 接近程度因子：越接近泊位，越抑制推进速度（防止冲过头）
    # 远离时因子接近 1（不阻断探索），接近时衰减
    near_factor = 1.0 / (1.0 + 0.6 * max(0.0, 2.0 - dist_cur))
    # 只在接触推箱时给推进奖励（避免小车空跑刷分）
    push_gate = 1.0 if contact > 0.5 else 0.0
    progress_reward = 3.0 * progress_delta * near_factor * push_gate

    # ---- 组件 B: 停靠对齐与低速（条件化，仅在接近泊位时激活）----
    # 接近程度门：dist < 1.5m 时开始生效，平滑过渡
    dock_proximity = max(0.0, min(1.0, (1.5 - dist_cur) / 1.5))
    # 低速因子：货箱速度越低越好（bounded，0~1）
    speed_factor = 1.0 / (1.0 + 4.0 * crate_speed)
    # 朝向对齐因子：货箱朝向与泊位朝向对齐（泊位朝向近似为 0，即 cos 接近 1）
    # 用 |cos| 的平滑形式，避免 atan2 分支
    crate_cos = obs[10]
    align_factor = max(0.0, crate_cos)  # 0~1，对齐时接近 1
    # 联合条件 proxy（几何平均，避免塌缩）
    settle_proxy = (speed_factor * align_factor) ** 0.5
    settle_reward = 1.5 * dock_proximity * settle_proxy

    # ---- 组件 C: 易碎冲击抑制（hinge，仅在接触且高速时生效）----
    # 接触时小车前向速度超过阈值才惩罚，避免误罚正常推箱
    impact_excess = max(0.0, cart_fwd_speed - 1.2)
    impact_penalty = -0.8 * impact_excess * contact

    total_reward = progress_reward + settle_reward + impact_penalty

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_settle_and_align": float(settle_reward),
        "fragile_impact_avoidance": float(impact_penalty),
    }

    return float(total_reward), components