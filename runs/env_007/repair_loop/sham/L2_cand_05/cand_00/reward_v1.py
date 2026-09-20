def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    _PREV_T = getattr(compute_reward, "_prev_t", None)
    if _PREV_T is None:
        compute_reward._prev_t = -1.0
        compute_reward._streak = 0
        compute_reward._paid = False
        compute_reward._entered = False

    t = float(next_obs[18])
    if t < compute_reward._prev_t or t <= 1.0 / 400.0:
        compute_reward._streak = 0
        compute_reward._paid = False
        compute_reward._entered = False
    compute_reward._prev_t = t

    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5

    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    cc = float(next_obs[10])
    cs = float(next_obs[11])
    if cc > 0.0:
        ang = cs
    else:
        ang = 1.0 if cs >= 0.0 else -1.0
    ang_err = abs(ang)

    in_tol = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = ang_err < 0.5
    slow = crate_speed < 0.05
    complete_state = in_tol and aligned and slow

    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = abs(cx)
    ay = abs(cy)
    bound_pen = 0.0
    if ax > 0.95:
        bound_pen -= 8.0 * (ax - 0.95)
    if ay > 0.95:
        bound_pen -= 8.0 * (ay - 0.95)

    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    excess = closing - 1.5
    if excess < 0.0:
        excess = 0.0
    gentleness = -2.0 * contact * excess

    if complete_state:
        compute_reward._streak += 1
    else:
        compute_reward._streak = 0

    success_event = 0.0
    if compute_reward._streak >= 10 and not compute_reward._paid:
        compute_reward._paid = True
        success_event = 300.0

    enter_event = 0.0
    if in_tol and aligned and not compute_reward._entered:
        compute_reward._entered = True
        enter_event = 5.0

    # ---- 主信号：持续接近度（每步为正，推箱靠近严格优于静止） ----
    proximity = 1.0 / (1.0 + 8.0 * dist)
    progress = 60.0 * proximity

    # 增量奖励（仅正向，作为加速收敛的附加项）
    delta = old_dist - dist
    if delta > 0.0:
        progress += 30.0 * delta

    # ---- 联合条件代理：几何平均，连续有梯度 ----
    f_pos = 1.0 / (1.0 + 40.0 * ((dx * dx + dy * dy) ** 0.5))
    f_align = 1.0 / (1.0 + 3.0 * ang_err)
    f_slow = 1.0 / (1.0 + 20.0 * crate_speed)
    dock_quality = 8.0 * (f_pos * f_align * f_slow) ** (1.0 / 3.0)

    # ---- 泊位附近速度惩罚：仅在 dist<0.15 时启用，避免压制推进 ----
    speed_pen = 0.0
    if dist < 0.15 and crate_speed > 0.05:
        speed_pen = -3.0 * (crate_speed - 0.05)

    total = progress + dock_quality + speed_pen + gentleness + bound_pen + success_event + enter_event

    components = {
        "progress": progress,
        "dock_quality": dock_quality,
        "speed_pen": speed_pen,
        "gentleness": gentleness,
        "bound_pen": bound_pen,
        "enter_event": enter_event,
        "success_event": success_event,
    }
    return float(total), components