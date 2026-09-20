# Response Record

```python
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

    # ---------- 货箱到泊位的有符号偏移（归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- 货箱速度（归一化 -> m/s） ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 货箱朝向误差 ----------
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    heading_err = abs(2.0 * (ch * 0.0 + 0.0))  # placeholder replaced below
    # 用叉积/点积求朝向误差绝对值（相对泊位朝向 0 弧度）
    heading_err = abs((sh * sh) ** 0.5)  # |sin| 近似；改用 atan2 形式
    import_free = True  # 说明：不使用 import
    # 精确朝向误差：|atan2(sh, ch)|，用 |sin| 与 |cos| 组合的连续代理
    if ch < 0.0:
        heading_err = 3.141592653589793 - abs((sh * sh) ** 0.5) if sh >= 0.0 else 3.141592653589793 - abs((sh * sh) ** 0.5)
    heading_err = abs((sh * sh) ** 0.5)

    # ---------- 完成条件（容差取自环境事实） ----------
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if heading_err < 0.5 else 0.0  # sin<0.5 约 <30°
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # ---------- 连续保持计数 ----------
    if complete_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 首次进入泊位（一次性） ----------
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 15.0

    # ---------- 组件 ----------
    components = {}

    # 1) 货箱向泊位推进（增量形式，只在更接近时给分）
    progress = old_dist - dist
    if progress < 0.0:
        progress = 0.0
    # 对齐门控：货箱朝向越接近泊位朝向，推进奖励越有效
    align_gate = 1.0 / (1.0 + 3.0 * heading_err)
    crate_to_dock_progress = 20.0 * progress * align_gate
    components["crate_to_dock_progress"] = crate_to_dock_progress

    # 2) 轻柔度：接触中且正在接近时惩罚
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["soft_contact"] = gentleness

    # 3) 接近泊位时的速度抑制（hinge，仅在接近时启用；作为惩罚）
    near = 1.0 / (1.0 + 8.0 * dist)
    speed_excess = crate_speed - 0.05
    if speed_excess < 0.0:
        speed_excess = 0.0
    dock_speed_penalty = -0.5 * near * speed_excess
    components["dock_speed_penalty"] = dock_speed_penalty

    # 4) 越界 hinge 惩罚（小车位置）
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    oob = 0.0
    if cx > 0.95:
        oob += (cx - 0.95)
    if cx < -0.95:
        oob += (-0.95 - cx)
    if cy > 0.95:
        oob += (cy - 0.95)
    if cy < -0.95:
        oob += (-0.95 - cy)
    out_of_bounds_penalty = -2.0 * oob
    components["out_of_bounds_penalty"] = out_of_bounds_penalty

    # 5) 一次性事件
    components["enter_dock_event"] = enter_event
    components["success_event"] = success_event

    # ---------- 完成状态下：除一次性事件外其余组件必须为 0 ----------
    if complete_state > 0.5:
        components["crate_to_dock_progress"] = 0.0
        components["soft_contact"] = 0.0
        components["dock_speed_penalty"] = 0.0
        components["out_of_bounds_penalty"] = 0.0

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```
