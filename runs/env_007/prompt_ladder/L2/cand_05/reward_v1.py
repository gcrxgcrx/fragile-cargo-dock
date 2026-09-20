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

    # ---------- 货箱到泊位偏移（归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 朝向误差（货箱朝向 vs 泊位朝向，泊位朝向取 +x 轴） ----------
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    # 货箱朝向角
    ang = 0.0
    if cc > 0.0:
        ang = cs
    else:
        ang = 1.0 if cs >= 0.0 else -1.0
    ang_err = abs(ang)  # 归一化误差代理，0=对齐

    # ---------- 完成条件（容差取自环境事实） ----------
    in_tol = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = ang_err < 0.5
    slow = crate_speed < 0.05
    complete_state = in_tol and aligned and slow

    # ---------- 边界守卫 ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = abs(cx)
    ay = abs(cy)
    bound_pen = 0.0
    if ax > 0.95:
        bound_pen -= 8.0 * (ax - 0.95)
    if ay > 0.95:
        bound_pen -= 8.0 * (ay - 0.95)

    # ---------- 轻柔度：接触时的接近速度惩罚 ----------
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -6.0 * contact * closing

    # ---------- 完成事件（一次性） ----------
    if complete_state:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if in_tol and aligned and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 5.0

    # ---------- 过程信号（完成状态下必须为 0） ----------
    progress = 0.0
    dock_quality = 0.0
    speed_pen = 0.0

    if not complete_state:
        # 增量式接近：只在更接近时给分
        delta = old_dist - dist
        if delta > 0.0:
            progress = 40.0 * delta

        # 泊位附近的低速门控（只惩罚，不给持续正分）
        if dist < 0.15:
            if crate_speed > 0.05:
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