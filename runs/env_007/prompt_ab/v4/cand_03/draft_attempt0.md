```python
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

    # ---------- 几何量（货箱到泊位的有符号偏移，归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 完成判据（严格取自环境事实：|dx|<=0.024, |dy|<=0.030, 朝向误差<30°, 速度<0.05 m/s）
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 货箱朝向误差
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    ang_err = abs(ccos - 1.0) + abs(csin)  # 0 表示完全对齐；<30° 时约 <0.27+0.5
    align_ok = 1.0 if (ccos >= 0.866) else 0.0  # cos30° = 0.866

    # 货箱速度（m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow_ok = 1.0 if crate_speed < 0.05 else 0.0

    # 完成条件是否满足（用于连续计数）
    cond = 1.0 if (inside > 0.5 and align_ok > 0.5 and slow_ok > 0.5) else 0.0
    if cond > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    done_state = 1.0 if (inside > 0.5 and align_ok > 0.5 and slow_ok > 0.5) else 0.0

    components = {}

    # ---------- 一次性事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if inside > 0.5 and align_ok > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    # ---------- 完成状态下：除一次性事件外全部为 0 ----------
    if done_state > 0.5:
        components["success_event"] = success_event
        components["enter_event"] = enter_event
        components["crate_progress"] = 0.0
        components["align_progress"] = 0.0
        components["gentleness"] = 0.0
        components["speed_penalty_near_dock"] = 0.0
        components["out_of_bounds"] = 0.0
        components["obstacle_penalty"] = 0.0
        components["action_cost"] = 0.0
        total = success_event + enter_event
        return (float(total), components)

    # ---------- 过程型信号（增量形式） ----------
    # 货箱到泊位的靠近增量
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    delta_dist = _PREV_DIST[0] - dist  # >0 表示更近了
    _PREV_DIST[0] = dist

    crate_progress = 6.0 * delta_dist

    # 对齐增量（门控乘在靠近增量上，避免持续收分）
    align_quality = 1.0 / (1.0 + 4.0 * ang_err)  # 0..1 连续
    align_progress = 2.0 * align_quality * max(0.0, delta_dist)

    # ---------- 轻柔度：唯一教"接近泊位减速"的信号 ----------
    cart_vx = float(next_obs[8]) * 3.0
    cart_vy = float(next_obs[9]) * 3.0
    crate_along_heading = cart_vx * float(obs[2]) + cart_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 接近泊位时的速度抑制（hinge，只在接近时启用） ----------
    near_dock = 1.0 if dist < 0.12 else 0.0
    speed_penalty_near_dock = 0.0
    if near_dock > 0.5:
        excess = crate_speed - 0.05
        if excess > 0.0:
            speed_penalty_near_dock = -2.0 * excess

    # ---------- 越界惩罚（hinge） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.95:
        oob += (abs(cart_x) - 0.95)
    if abs(cart_y) > 0.95:
        oob += (abs(cart_y) - 0.95)
    out_of_bounds = -5.0 * oob

    # ---------- 障碍接近惩罚（hinge） ----------
    sf = float(next_obs[15])
    sl = float(next_obs[16])
    sr = float(next_obs[17])
    obst = 0.0
    if sf > 0.85:
        obst += (sf - 0.85)
    if sl > 0.85:
        obst += (sl - 0.85)
    if sr > 0.85:
        obst += (sr - 0.85)
    obstacle_penalty = -1.0 * obst

    # ---------- 动作代价（轻量） ----------
    a0 = float(action[0])
    a1 = float(action[1])
    action_cost = -0.01 * (a0 * a0 + a1 * a1)

    components["success_event"] = success_event
    components["enter_event"] = enter_event
    components["crate_progress"] = crate_progress
    components["align_progress"] = align_progress
    components["gentleness"] = gentleness
    components["speed_penalty_near_dock"] = speed_penalty_near_dock
    components["out_of_bounds"] = out_of_bounds
    components["obstacle_penalty"] = obstacle_penalty
    components["action_cost"] = action_cost

    total = (
        success_event
        + enter_event
        + crate_progress
        + align_progress
        + gentleness
        + speed_penalty_near_dock
        + out_of_bounds
        + obstacle_penalty
        + action_cost
    )
    return (float(total), components)
```