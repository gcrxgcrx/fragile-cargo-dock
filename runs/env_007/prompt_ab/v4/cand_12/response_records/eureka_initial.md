# Response Record

```python
# 模块级状态：用于一次性事件与回合边界检测
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# 泊位容差（取自环境事实：|obs[12]| <= 0.024, |obs[13]| <= 0.030）
_DOCK_X_TOL = 0.024
_DOCK_Y_TOL = 0.030
_ALIGN_TOL = 0.5          # cos(30°) = 0.866  -> 用角度误差 < 30°
_SPEED_TOL = 0.05 / 3.0   # 货箱速度 0.05 m/s，obs 单位是 /3.0


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------- 几何量 ----------
    dx = float(next_obs[12])   # 货箱到泊位有符号 x 偏移
    dy = float(next_obs[13])   # 货箱到泊位有符号 y 偏移
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12])
    ody = float(obs[13])
    odist = (odx * odx + ody * ody) ** 0.5

    # 完成状态判据
    in_dock = (abs(dx) <= _DOCK_X_TOL) and (abs(dy) <= _DOCK_Y_TOL)

    # 朝向对齐：货箱朝向误差
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    align_cos = cc  # 与泊位（世界系）对齐，cos 接近 1
    aligned = align_cos > 0.866  # < 30°

    # 货箱速度
    cvx = float(next_obs[8])
    cvy = float(next_obs[9])
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = crate_speed < _SPEED_TOL

    # ---------- 完成条件与连续计数 ----------
    cond = in_dock and aligned and slow
    if cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 一次性事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    first_enter_event = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        first_enter_event = 20.0

    # 完成状态下：除一次性事件外其余组件必须恰好为 0
    if in_dock and aligned and slow:
        components["success_event"] = success_event
        components["first_enter_event"] = first_enter_event
        return (float(success_event + first_enter_event), components)

    # ---------- 过程型信号（仅在未完成区域生效） ----------

    # 1) 增量式接近：只在"这一帧更接近"时给分
    progress = odist - dist
    if progress < 0.0:
        progress = 0.0
    # 用对齐度作为门控乘在增量上（对齐好时，推进更有价值）
    align_gate = 0.5 + 0.5 * max(0.0, align_cos)
    components["crate_to_dock_progress"] = 2.0 * progress * align_gate

    # 2) 首次进入泊位（一次性，已在上方计算）
    components["first_enter_event"] = first_enter_event

    # 3) 轻柔度：接触时惩罚接近速度（唯一教会减速的信号）
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    components["gentleness"] = -0.05 * contact * closing

    # 4) 接近泊位时的速度抑制（hinge，仅在接近泊位时启用）
    near_dock_gate = 1.0
    if dist > 0.15:
        near_dock_gate = 0.0
    elif dist > 0.05:
        near_dock_gate = (0.15 - dist) / 0.10
    excess_speed = crate_speed - _SPEED_TOL
    if excess_speed < 0.0:
        excess_speed = 0.0
    components["crate_speed_near_dock"] = -3.0 * near_dock_gate * excess_speed

    # 5) 越界惩罚（小车 / 货箱接近仓库边界）
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.85:
        oob += (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        oob += (abs(cart_y) - 0.85)
    components["out_of_bounds"] = -2.0 * oob

    # 6) 障碍接近惩罚（前向传感器）
    sf = float(next_obs[15])
    if sf > 0.8:
        components["obstacle"] = -0.5 * (sf - 0.8)
    else:
        components["obstacle"] = 0.0

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```
