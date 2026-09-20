def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    state = getattr(compute_reward, "_state", None)
    if state is None:
        state = {"prev_t": -1.0, "streak": 0, "paid": False, "entered": False}
        compute_reward._state = state

    t = float(next_obs[18])
    if t < state["prev_t"] or t <= 1.0 / 400.0:
        state["streak"] = 0
        state["paid"] = False
        state["entered"] = False
    state["prev_t"] = t

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
    crate_ang = 0.0
    if cc > 0.0:
        crate_ang = cs
    else:
        crate_ang = 1.0 if cs >= 0.0 else -1.0
    ang_err = abs(crate_ang)

    in_tol = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = ang_err < 0.5
    slow = crate_speed < 0.05
    complete_state = in_tol and aligned and slow

    # ---------- 边界：平滑门控而非主导惩罚 ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = abs(cx)
    ay = abs(cy)
    bound_pen = 0.0
    if ax > 0.9:
        bound_pen -= 4.0 * (ax - 0.9)
    if ay > 0.9:
        bound_pen -= 4.0 * (ay - 0.9)

    # ---------- 轻柔度：只在接触且接近时轻量惩罚 ----------
    crate_along = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -1.5 * contact * closing

    # ---------- 完成事件 ----------
    if complete_state:
        state["streak"] += 1
    else:
        state["streak"] = 0

    success_event = 0.0
    if state["streak"] >= 10 and not state["paid"]:
        state["paid"] = True
        success_event = 300.0

    enter_event = 0.0
    if in_tol and aligned and not state["entered"]:
        state["entered"] = True
        enter_event = 8.0

    # ---------- 主进度：凸化连续接近 + 增量 ----------
    progress = 0.0
    if not complete_state:
        # 凸化接近：距离越近，梯度越强
        proximity = 1.0 / (1.0 + 6.0 * dist)
        progress += 6.0 * proximity * proximity
        delta = old_dist - dist
        if delta > 0.0:
            progress += 25.0 * delta

    # ---------- 泊位质量：位置 × 朝向 × 低速 的几何平均（连续） ----------
    dock_quality = 0.0
    if not complete_state:
        pos_f = max(0.0, 1.0 - dist / 0.25)
        ori_f = max(0.0, 1.0 - ang_err / 1.2)
        spd_f = max(0.0, 1.0 - crate_speed / 0.6)
        if pos_f > 0.0 and ori_f > 0.0 and spd_f > 0.0:
            dock_quality = 8.0 * (pos_f * ori_f * spd_f) ** (1.0 / 3.0)

    # ---------- 泊位附近低速门控惩罚（仅近泊位且过快时） ----------
    speed_pen = 0.0
    if not complete_state and dist < 0.12 and crate_speed > 0.05:
        speed_pen = -2.0 * (crate_speed - 0.05)

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