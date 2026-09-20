_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 基础量 ----------
    # 货箱到泊位中心的偏移（归一化）
    ex = float(next_obs[12])
    ey = float(next_obs[13])
    dist = (ex * ex + ey * ey) ** 0.5

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（相对泊位对齐，即相对世界 x 轴对齐）
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    if ch > 1.0:
        ch = 1.0
    if ch < -1.0:
        ch = -1.0
    heading_err = abs(sh)  # |sin(theta)| 与角度误差单调同向，30° 时约 0.5

    # 接触与接近速度代理
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0

    # ---------- 完成条件（严格取自环境事实） ----------
    inside = 1.0 if (abs(ex) <= 0.024 and abs(ey) <= 0.030) else 0.0
    aligned = 1.0 if heading_err < 0.5 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    cond = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if cond > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 组件 1：货箱接近泊位的增量进度 ----------
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    delta = _PREV_DIST[0] - dist          # >0 表示这一帧更接近
    _PREV_DIST[0] = dist
    if delta > 0.05:
        delta = 0.05
    if delta < -0.05:
        delta = -0.05
    crate_progress = 20.0 * delta

    # ---------- 组件 2：接近泊位时的速度抑制（门控，仅在靠近时激活） ----------
    near_gate = 1.0 if dist < 0.10 else 0.0
    speed_near = -2.0 * near_gate * crate_speed

    # ---------- 组件 3：接触轻柔度 ----------
    gentleness = -0.05 * contact * closing

    # ---------- 组件 4：朝向对齐 shaping（仅在泊位附近激活） ----------
    align_term = -0.3 * near_gate * heading_err

    # ---------- 组件 5：越界 hinge 惩罚 ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.95:
        oob -= 1.0 * (abs(cart_x) - 0.95)
    if abs(cart_y) > 0.95:
        oob -= 1.0 * (abs(cart_y) - 0.95)
    obstacle = float(next_obs[15])
    if obstacle > 0.9:
        oob -= 0.5 * (obstacle - 0.9)
    out_of_bounds = oob

    # ---------- 组件 6：动作平滑（轻量） ----------
    smoothness = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 组件 7：首次进入泊位（一次性） ----------
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 30.0

    # ---------- 组件 8：完成事件（一次性） ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    total = (crate_progress + speed_near + gentleness + align_term
             + out_of_bounds + smoothness + enter_event + success_event)

    components = {
        "crate_progress": crate_progress,
        "speed_near_dock": speed_near,
        "gentleness": gentleness,
        "align_near_dock": align_term,
        "out_of_bounds": out_of_bounds,
        "action_smoothness": smoothness,
        "enter_dock_event": enter_event,
        "success_event": success_event,
    }
    return float(total), components