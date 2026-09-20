_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    components = {}

    # ---------- 几何量（米制近似，半宽 5.0，半高 4.0） ----------
    dx = float(next_obs[12]) * 5.0
    dy = float(next_obs[13]) * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12]) * 5.0
    ody = float(obs[13]) * 4.0
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- 完成条件（严格取自环境事实容差） ----------
    in_dock = 1.0 if (abs(float(next_obs[12])) <= 0.024 and abs(float(next_obs[13])) <= 0.030) else 0.0

    # 朝向误差
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    ang = ccos * 1.0 + csin * 0.0
    if ang > 1.0:
        ang = 1.0
    if ang < -1.0:
        ang = -1.0
    ang_err = (1.0 - ang) ** 0.5 * 1.414213562 * 0.5 + (1.0 - ang) * 0.5  # 近似 |angle|
    align = 1.0 if ang_err < 0.5235987756 else 0.0

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    completed_state = 1.0 if (in_dock > 0.5 and align > 0.5 and slow > 0.5) else 0.0

    # 连续计数
    if completed_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 接触轻柔度 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -1.0 * contact * closing

    # ---------- 越界守卫 ----------
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))
    over_x = cart_x - 0.95
    if over_x < 0.0:
        over_x = 0.0
    over_y = cart_y - 0.95
    if over_y < 0.0:
        over_y = 0.0
    bound_pen = -10.0 * (over_x * over_x + over_y * over_y) - 5.0 * (over_x + over_y)

    # 货箱越界近似：货箱世界坐标
    crx = float(next_obs[6]) * 3.0
    cry = float(next_obs[7]) * 3.0
    ch = float(next_obs[2])
    sh = float(next_obs[3])
    crate_wx = float(next_obs[0]) * 5.0 + crx * ch - cry * sh
    crate_wy = float(next_obs[1]) * 4.0 + crx * sh + cry * ch
    crate_bx = abs(crate_wx) / 5.0
    crate_by = abs(crate_wy) / 4.0
    cover_x = crate_bx - 0.95
    if cover_x < 0.0:
        cover_x = 0.0
    cover_y = crate_by - 0.95
    if cover_y < 0.0:
        cover_y = 0.0
    crate_bound_pen = -10.0 * (cover_x * cover_x + cover_y * cover_y) - 5.0 * (cover_x + cover_y)

    # ---------- 主进展：增量式靠近（门控对齐） ----------
    progress = 0.0
    if completed_state < 0.5:
        delta = old_dist - dist
        if delta > 0.0:
            align_gate = 0.25 + 0.75 * max(0.0, ang)
            progress = 20.0 * delta * align_gate
            if progress > 2.0:
                progress = 2.0
        else:
            progress = 2.0 * delta  # 远离时轻微惩罚（负值）

    # 接近泊位时的速度抑制（仅在未完成且接近时）
    speed_near_pen = 0.0
    if completed_state < 0.5 and dist < 1.5:
        gate = (1.5 - dist) / 1.5
        if crate_speed > 0.05:
            speed_near_pen = -3.0 * gate * (crate_speed - 0.05)

    # ---------- 一次性事件 ----------
    enter_event = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 30.0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 完成状态下清零所有持续项 ----------
    if completed_state > 0.5:
        progress = 0.0
        speed_near_pen = 0.0
        gentleness = 0.0
        bound_pen = 0.0
        crate_bound_pen = 0.0

    components["progress"] = progress
    components["gentleness"] = gentleness
    components["speed_near_dock"] = speed_near_pen
    components["bound_guard"] = bound_pen + crate_bound_pen
    components["enter_event"] = enter_event
    components["success_event"] = success_event

    total = progress + gentleness + speed_near_pen + bound_pen + crate_bound_pen + enter_event + success_event
    return (float(total), components)