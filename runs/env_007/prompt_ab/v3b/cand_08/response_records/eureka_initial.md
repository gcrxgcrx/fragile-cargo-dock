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

    # ---------- 货箱到泊位偏移（归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 泊位容差：|dx| <= 0.024, |dy| <= 0.030
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 朝向误差 ----------
    cos_err = float(next_obs[10])
    sin_err = float(next_obs[11])
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    heading_err = abs(sin_err)  # |sin| 近似小角度误差，30° -> 0.5

    # ---------- 组件 1：货箱向泊位推进（增量形式） ----------
    progress = 0.0
    if _PREV_DIST[0] >= 0.0:
        progress = (_PREV_DIST[0] - dist) * 12.0
        if progress > 0.6:
            progress = 0.6
        if progress < -0.6:
            progress = -0.6
    _PREV_DIST[0] = dist

    # ---------- 组件 2：轻柔接触（唯一能教减速的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 3：泊位内对齐与静止（仅在泊位内激活的温和 shaping） ----------
    dock_quality = 0.0
    if inside > 0.5:
        align_f = 1.0 - heading_err / 0.5
        if align_f < 0.0:
            align_f = 0.0
        speed_f = 1.0 - crate_speed / 0.05
        if speed_f < 0.0:
            speed_f = 0.0
        dock_quality = 0.5 * align_f * speed_f

    # ---------- 组件 4：边界 hinge 惩罚 ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    boundary = 0.0
    for v in (cx, cy):
        a = abs(v)
        if a > 0.90:
            boundary -= 0.5 * (a - 0.90)
    for v in (dx, dy):
        a = abs(v)
        if a > 0.45:
            boundary -= 0.5 * (a - 0.45)

    # ---------- 组件 5：动作平滑（轻量） ----------
    smooth = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 完成条件与连续计数 ----------
    cond = (inside > 0.5) and (heading_err < 0.5) and (crate_speed < 0.05)
    if cond:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    total = (
        progress
        + gentleness
        + dock_quality
        + boundary
        + smooth
        + enter_bonus
        + success_event
    )

    components = {
        "crate_to_dock_progress": progress,
        "soft_contact_gentleness": gentleness,
        "dock_quality": dock_quality,
        "boundary_penalty": boundary,
        "action_smoothness": smooth,
        "enter_dock_bonus": enter_bonus,
        "success_event": success_event,
    }
    return float(total), components
```
