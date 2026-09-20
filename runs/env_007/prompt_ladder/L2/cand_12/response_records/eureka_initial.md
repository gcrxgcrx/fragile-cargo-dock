# Response Record

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

    components = {}

    # ---------- 几何常量（来自环境事实） ----------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])

    # 归一化距离度量（用于增量推进）
    dist = (dx * dx + dy * dy) ** 0.5

    # ---------- 完成条件（显式从 obs 推断） ----------
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 朝向误差：货箱朝向 vs 泊位朝向（泊位朝向假定与 +x 轴一致，误差用 cos 衡量）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差 < 30° 等价于 cos(误差) > cos(30°) ≈ 0.866
    align = 1.0 if crate_cos > 0.866 else 0.0

    # 货箱速度（世界系，恢复米/秒）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    done_state = 1.0 if (inside > 0.5 and align > 0.5 and slow > 0.5) else 0.0

    if done_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 越界守卫（单调下降，|x|>0.95 明显生效） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    ax = abs(cart_x)
    ay = abs(cart_y)
    bound_x = 0.0
    bound_y = 0.0
    if ax > 0.95:
        bound_x = (ax - 0.95) / 0.10
        if bound_x > 1.0:
            bound_x = 1.0
    if ay > 0.95:
        bound_y = (ay - 0.95) / 0.10
        if bound_y > 1.0:
            bound_y = 1.0
    bound_pen = -8.0 * (bound_x * bound_x + bound_y * bound_y)
    components["out_of_bounds"] = bound_pen

    # ---------- 接触轻柔度（唯一教会减速的信号） ----------
    crate_along = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -1.2 * contact * closing
    components["gentleness"] = gentleness

    # ---------- 完成状态：除一次性事件外其余组件必须恰好为 0 ----------
    if done_state > 0.5:
        success_event = 0.0
        if _STREAK[0] >= 10 and not _PAID[0]:
            _PAID[0] = True
            success_event = 300.0
        components["success_event"] = success_event
        components["progress"] = 0.0
        components["first_enter"] = 0.0
        components["speed_gate"] = 0.0
        total = bound_pen + gentleness + success_event
        return (float(total), components)

    # ---------- 推进增量（只在更接近时给分，避免悬停收割） ----------
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    delta = _PREV_DIST[0] - dist
    _PREV_DIST[0] = dist
    if delta < 0.0:
        delta = 0.0
    progress = 12.0 * delta
    components["progress"] = progress

    # ---------- 首次进入泊位（一次性） ----------
    first_enter = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        first_enter = 40.0
    components["first_enter"] = first_enter

    # ---------- 接近泊位时的速度门控惩罚（仅惩罚，不作正项） ----------
    # 货箱接近泊位（dist < 0.10）且速度过快时惩罚
    speed_gate = 0.0
    if dist < 0.10 and crate_speed > 0.05:
        over = crate_speed - 0.05
        speed_gate = -6.0 * over * over
    components["speed_gate"] = speed_gate

    total = bound_pen + gentleness + progress + first_enter + speed_gate
    return (float(total), components)
```
