```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode 边界检测 ----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    components = {}

    # ---- 几何量：货箱到泊位偏移（归一化），恢复为米制 ----
    dx = float(next_obs[12]) * 5.0
    dy = float(next_obs[13]) * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    # ---- 完成条件（显式从 obs 推断，容差取自环境事实）----
    inside = 1.0 if (abs(float(next_obs[12])) <= 0.024 and abs(float(next_obs[13])) <= 0.030) else 0.0
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 朝向误差：货箱朝向 vs 泊位朝向（泊位朝向取 +x 轴，误差 < 30° 即对齐）
    crate_ang = 0.0
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    crate_ang = (sh * sh + ch * ch) ** 0.5
    if crate_ang > 1e-6:
        cos_err = ch / crate_ang
    else:
        cos_err = 1.0
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    # 朝向误差角（弧度），用 cos 反推：误差 < 30° 时 cos > 0.866
    aligned = 1.0 if cos_err >= 0.866 else 0.0

    slow = 1.0 if crate_speed < 0.05 else 0.0

    done_now = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if done_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---- 完成状态：除一次性事件外所有组件必须恰好为 0 ----
    if done_now > 0.5:
        success_event = 0.0
        if _STREAK[0] >= 10 and not _PAID[0]:
            _PAID[0] = True
            success_event = 300.0
        components["success_event"] = success_event
        components["enter_dock_event"] = 0.0
        components["crate_progress"] = 0.0
        components["gentleness"] = 0.0
        components["cart_bounds"] = 0.0
        components["crate_bounds"] = 0.0
        components["dock_speed_penalty"] = 0.0
        _PREV_DIST[0] = dist
        total = success_event
        return (float(total), components)

    # ---- 1) 货箱向泊位推进（增量形式，避免悬停收割）----
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    progress = _PREV_DIST[0] - dist
    if progress < 0.0:
        progress = 0.0
    # 对齐门控：只有货箱朝向大致对齐时推进才给满分
    align_gate = 0.5 + 0.5 * cos_err
    if align_gate < 0.0:
        align_gate = 0.0
    if align_gate > 1.0:
        align_gate = 1.0
    crate_progress = 2.0 * progress * align_gate
    _PREV_DIST[0] = dist

    # ---- 2) 轻柔度：接触时惩罚接近速度 ----
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -1.0 * contact * closing

    # ---- 3) 接近泊位时抑制货箱速度（仅惩罚，不奖励静止）----
    near = 1.0 if dist < 0.5 else 0.0
    dock_speed_penalty = 0.0
    if near > 0.5:
        over = crate_speed - 0.05
        if over > 0.0:
            dock_speed_penalty = -2.0 * over * over

    # ---- 4) 越界守卫：小车位置 ----
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    cart_bounds = 0.0
    if cx > 0.95:
        cart_bounds -= 5.0 * (cx - 0.95)
    if cy > 0.95:
        cart_bounds -= 5.0 * (cy - 0.95)

    # ---- 5) 越界守卫：货箱位置（由相对位置恢复世界坐标）----
    rx = float(next_obs[6]) * 3.0
    ry = float(next_obs[7]) * 3.0
    ch2 = float(obs[2])
    sh2 = float(obs[3])
    crate_wx = rx * ch2 - ry * sh2 + float(next_obs[0]) * 5.0
    crate_wy = rx * sh2 + ry * ch2 + float(next_obs[1]) * 4.0
    cwx = abs(crate_wx) / 5.0
    cwy = abs(crate_wy) / 4.0
    crate_bounds = 0.0
    if cwx > 0.95:
        crate_bounds -= 5.0 * (cwx - 0.95)
    if cwy > 0.95:
        crate_bounds -= 5.0 * (cwy - 0.95)

    # ---- 6) 首次进入泊位（一次性，整局只发一次）----
    enter_dock_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_dock_event = 20.0

    components["crate_progress"] = crate_progress
    components["gentleness"] = gentleness
    components["dock_speed_penalty"] = dock_speed_penalty
    components["cart_bounds"] = cart_bounds
    components["crate_bounds"] = crate_bounds
    components["enter_dock_event"] = enter_dock_event
    components["success_event"] = 0.0

    total = (crate_progress + gentleness + dock_speed_penalty +
             cart_bounds + crate_bounds + enter_dock_event)
    return (float(total), components)
```