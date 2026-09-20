_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 几何量（由 obs 声明恢复） ----------
    # 泊位半宽/半高（obs[12]/obs[13] 为归一化偏移）
    # 容差：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 朝向误差（货箱朝向 vs 泊位朝向，泊位朝向视为 0）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 货箱朝向角
    # 用 cos 直接度量对齐：|angle| < 30° => cos > cos30 = 0.866
    align_cos = crate_cos  # 若泊位目标朝向为 +x
    if align_cos < 0.0:
        align_cos = 0.0

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 完成条件（严格按环境事实）
    inside = (dx <= 0.024 and dx >= -0.024 and dy <= 0.030 and dy >= -0.030)
    aligned = (crate_cos >= 0.866)
    slow = (crate_speed < 0.05)
    cond_ok = inside and aligned and slow

    # ---------- 连续计数 ----------
    if cond_ok:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 一次性事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if inside and aligned and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    # ---------- 完成状态下：除一次性事件外全部为 0 ----------
    if cond_ok:
        components = {
            "crate_to_dock_progress": 0.0,
            "crate_align_progress": 0.0,
            "soft_contact": 0.0,
            "speed_penalty_near_dock": 0.0,
            "out_of_bounds": 0.0,
            "action_smooth": 0.0,
            "enter_event": enter_event,
            "success_event": success_event,
        }
        total = enter_event + success_event
        return (float(total), components)

    # ---------- 过程型信号（增量 / 门控 / 惩罚） ----------
    # 1) 货箱到泊位距离的增量（只在更接近时给正分）
    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5
    progress = old_dist - dist  # >0 表示靠近
    if progress < 0.0:
        progress = 0.0
    crate_to_dock_progress = 30.0 * progress

    # 2) 朝向对齐的增量门控（对齐度 × 本帧靠近增量）
    # 对齐因子：cos 在 [0,1]，仅作为门控乘在靠近增量上
    align_gate = align_cos
    crate_align_progress = 8.0 * align_gate * progress

    # 3) 轻柔接触惩罚（唯一教会"接近泊位减速"的信号）
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # 4) 接近泊位时的速度抑制（hinge，仅在接近时启用；作为惩罚非正项）
    # 距离阈值：dist 归一化，取 0.15 作为"接近"门限
    near_gate = 0.0
    if dist < 0.15:
        near_gate = 1.0 - dist / 0.15
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    speed_penalty_near_dock = -2.0 * near_gate * speed_excess

    # 5) 越界惩罚（hinge，仅在接近边界时生效）
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    out_of_bounds = 0.0
    # 小车边界（归一化坐标，|x|,|y| 接近 1 时惩罚）
    if cart_x > 0.92:
        out_of_bounds -= 2.0 * (cart_x - 0.92)
    if cart_x < -0.92:
        out_of_bounds -= 2.0 * (-cart_x - 0.92)
    if cart_y > 0.92:
        out_of_bounds -= 2.0 * (cart_y - 0.92)
    if cart_y < -0.92:
        out_of_bounds -= 2.0 * (-cart_y - 0.92)
    # 货箱边界（由 obs[12]/obs[13] 相对泊位，泊位在远侧；这里用泊位偏移的极端值近似）
    if dx > 1.0:
        out_of_bounds -= 2.0 * (dx - 1.0)
    if dx < -1.0:
        out_of_bounds -= 2.0 * (-dx - 1.0)
    if dy > 1.0:
        out_of_bounds -= 2.0 * (dy - 1.0)
    if dy < -1.0:
        out_of_bounds -= 2.0 * (-dy - 1.0)

    # 6) 动作平滑（轻量，不压制推动）
    action_smooth = -0.02 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_align_progress": crate_align_progress,
        "soft_contact": gentleness,
        "speed_penalty_near_dock": speed_penalty_near_dock,
        "out_of_bounds": out_of_bounds,
        "action_smooth": action_smooth,
        "enter_event": enter_event,
        "success_event": success_event,
    }

    total = (
        crate_to_dock_progress
        + crate_align_progress
        + gentleness
        + speed_penalty_near_dock
        + out_of_bounds
        + action_smooth
        + enter_event
        + success_event
    )

    return (float(total), components)