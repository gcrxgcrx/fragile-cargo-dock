def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 回合边界检测（obs[18] 单调递增，重置时回落）----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---- 1. 货箱到泊位进度（增量形式，避免悬停收割）----
    # 泊位几何：|obs[12]| <= 0.024, |obs[13]| <= 0.030 为完全进入容差
    # 用归一化偏移的欧氏距离作为进度度量
    d_old = (obs[12] ** 2 + obs[13] ** 2) ** 0.5
    d_new = (next_obs[12] ** 2 + next_obs[13] ** 2) ** 0.5
    progress = d_old - d_new                      # 只在这一帧更接近时为正
    crate_progress = 2.0 * progress
    components["crate_progress"] = crate_progress

    # ---- 2. 进入泊位的门控（接近泊位时激活辅助 shaping）----
    # 用归一化偏移构造 0~1 的接近度门，仅在接近泊位时启用
    near_gate = 1.0 / (1.0 + 40.0 * d_new)        # d_new 小 -> 接近 1
    if near_gate > 1.0:
        near_gate = 1.0

    # ---- 3. 泊位内对齐 + 静止 shaping（门控后的小额过程信号）----
    # 朝向误差：货箱朝向与泊位期望朝向（0 度）的夹角
    crate_angle_err = abs(next_obs[11])            # sin(heading)，|err|<30° 时 |sin|<=0.5
    align_factor = 1.0 - crate_angle_err / 0.5
    if align_factor < 0.0:
        align_factor = 0.0
    if align_factor > 1.0:
        align_factor = 1.0

    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5
    # 静止因子：速度 < 0.05 m/s 时接近 1
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    if speed_factor > 1.0:
        speed_factor = 1.0

    # 联合质量（几何平均，避免塌缩）
    dock_quality = (align_factor * speed_factor) ** 0.5
    dock_shaping = 0.5 * near_gate * dock_quality
    components["dock_quality"] = dock_shaping

    # ---- 4. 轻柔接触信号（核心技能信号）----
    # 货箱速度沿车头方向的分量
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["gentleness"] = gentleness

    # ---- 5. 边界 hinge 惩罚（小车 + 货箱）----
    # 小车位置归一化到 [-1,1]，接近边界时惩罚
    cart_margin = 0.85
    cart_pen = 0.0
    ax = obs[0]
    ay = obs[1]
    if ax > cart_margin:
        cart_pen += ax - cart_margin
    if ax < -cart_margin:
        cart_pen += -cart_margin - ax
    if ay > cart_margin:
        cart_pen += ay - cart_margin
    if ay < -cart_margin:
        cart_pen += -cart_margin - ay
    # 货箱到泊位偏移过大（远离仓库中心区域）时轻微惩罚
    crate_out = 0.0
    if d_new > 1.2:
        crate_out = (d_new - 1.2) * 0.5
    components["out_of_bounds"] = -1.0 * cart_pen - 1.0 * crate_out

    # ---- 6. 动作平滑（轻量，不压制推动）----
    smooth = -0.01 * (action[0] ** 2 + action[1] ** 2)
    components["action_smooth"] = smooth

    # ---- 7. 完成事件（一次性）----
    # 完成条件：完全进入 + 朝向对齐 < 30° + 速度 < 0.05 m/s
    inside = 1.0 if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030) else 0.0
    aligned = 1.0 if crate_angle_err <= 0.5 else 0.0
    still = 1.0 if crate_speed < 0.05 else 0.0
    if inside > 0.5 and aligned > 0.5 and still > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---- 8. 首次进入泊位奖励（一次性）----
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0
    components["enter_event"] = enter_event

    total = (crate_progress + dock_shaping + gentleness
             + components["out_of_bounds"] + smooth
             + success_event + enter_event)
    return float(total), components