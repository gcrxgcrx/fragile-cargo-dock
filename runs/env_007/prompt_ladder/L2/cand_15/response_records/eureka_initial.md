# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_D = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_D[0] = -1.0
    _PREV_T[0] = t

    # ---------- 泊位几何（来自环境事实） ----------
    # 泊位容差: |obs[12]| <= 0.024, |obs[13]| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # 货箱朝向误差（相对泊位朝向，泊位朝向由 cos/sin 给出）
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差用 sin 分量近似（误差角 sin）
    heading_err = abs(crate_sin)

    # 货箱速度
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 完成条件（严格取自环境事实）
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    aligned = 1.0 if heading_err < 0.5 else 0.0   # sin(30°) = 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0
    done_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # ---------- 连续计数 ----------
    if done_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 接触轻柔度代理 ----------
    cart_cos = float(obs[2])
    cart_sin = float(obs[3])
    crate_along_heading = cvx * cart_cos + cvy * cart_sin
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -1.2 * contact * closing

    # ---------- 越界守卫 ----------
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    # 货箱世界位置恢复
    rel_x = float(next_obs[6]) * 3.0
    rel_y = float(next_obs[7]) * 3.0
    crate_wx = rel_x * cart_cos - rel_y * cart_sin + float(next_obs[0]) * 5.0
    crate_wy = rel_x * cart_sin + rel_y * cart_cos + float(next_obs[1]) * 4.0
    # 归一化到半宽/半高
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0

    bound_pen = 0.0
    if cx > 0.95:
        bound_pen -= 3.0 * (cx - 0.95) * 10.0
    if cy > 0.95:
        bound_pen -= 3.0 * (cy - 0.95) * 10.0
    acx = abs(crate_nx)
    acy = abs(crate_ny)
    if acx > 0.95:
        bound_pen -= 3.0 * (acx - 0.95) * 10.0
    if acy > 0.95:
        bound_pen -= 3.0 * (acy - 0.95) * 10.0

    # ---------- 完成状态下其余组件必须为 0 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位（一次性）
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 15.0

    # 过程型组件（在完成状态下必须为 0）
    progress = 0.0
    align_gate = 0.0
    speed_gate = 0.0

    if done_state < 0.5:
        # 增量式接近信号：只在更接近时给分
        delta = prev_dist - dist
        if delta > 0.0:
            progress = 20.0 * delta

        # 对齐门控乘在接近增量上（不单独给分）
        if delta > 0.0:
            align_factor = max(0.0, 1.0 - heading_err / 0.5)
            align_gate = 5.0 * delta * align_factor

        # 接近泊位时速度过快惩罚（hinge，仅在接近时激活）
        if dist < 0.15:
            if crate_speed > 0.05:
                speed_gate = -2.0 * (crate_speed - 0.05)

    components = {
        "crate_to_dock_progress": progress,
        "align_gated_progress": align_gate,
        "speed_penalty_near_dock": speed_gate,
        "gentleness": gentleness,
        "out_of_bounds": bound_pen,
        "enter_event": enter_event,
        "success_event": success_event,
    }

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components
```
