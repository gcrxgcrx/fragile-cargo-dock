# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [None]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = None
    _PREV_T[0] = t

    # ---------- 基础量 ----------
    # 货箱到泊位的偏移（归一化），恢复为米制
    dx = float(next_obs[12]) * 5.0
    dy = float(next_obs[13]) * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    # 货箱速度（世界系，米/秒）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    ang_err = (sh * sh) ** 0.5  # |sin(theta)|，theta 为货箱相对泊位朝向误差
    if ang_err > 1.0:
        ang_err = 1.0

    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # ---------- 1) 主进展：货箱到泊位距离的增量（米） ----------
    progress = 0.0
    if _PREV_DIST[0] is not None:
        progress = _PREV_DIST[0] - dist
        if progress > 0.5:
            progress = 0.5
        if progress < -0.5:
            progress = -0.5
    _PREV_DIST[0] = dist
    r_progress = 20.0 * progress

    # ---------- 2) 轻柔接触 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    r_gentle = -0.05 * contact * closing

    # ---------- 3) 泊位内速度抑制（门控：仅在接近泊位时激活） ----------
    # 接近度门控：dist 在 0.30 m 内才启用
    if dist < 0.30:
        gate = 1.0 - dist / 0.30
        if gate < 0.0:
            gate = 0.0
        speed_excess = crate_speed - 0.05
        if speed_excess < 0.0:
            speed_excess = 0.0
        r_dock_speed = -2.0 * gate * speed_excess
    else:
        r_dock_speed = 0.0

    # ---------- 4) 朝向对齐 shaping（仅在泊位附近） ----------
    if dist < 0.40:
        r_align = -1.0 * (1.0 - dist / 0.40) * ang_err
    else:
        r_align = 0.0

    # ---------- 5) 越界 hinge 惩罚 ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    r_bounds = 0.0
    if abs(cx) > 0.95:
        r_bounds -= 2.0 * (abs(cx) - 0.95)
    if abs(cy) > 0.95:
        r_bounds -= 2.0 * (abs(cy) - 0.95)

    # ---------- 6) 完成事件：连续 10 步满足条件，一次性奖励 ----------
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    aligned = ang_err < 0.5  # sin(30°) = 0.5
    slow = crate_speed < 0.05
    if inside and aligned and slow:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # 首次进入泊位：一次性奖励
    r_entered = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        r_entered = 30.0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---------- 7) 动作平滑（轻量，可选） ----------
    r_smooth = -0.005 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    total = (
        r_progress
        + r_gentle
        + r_dock_speed
        + r_align
        + r_bounds
        + r_entered
        + success_event
        + r_smooth
    )

    components = {
        "crate_to_dock_progress": r_progress,
        "soft_contact_penalty": r_gentle,
        "dock_speed_suppression": r_dock_speed,
        "orientation_alignment": r_align,
        "out_of_bounds_penalty": r_bounds,
        "first_enter_dock_bonus": r_entered,
        "success_event": success_event,
        "action_smoothness": r_smooth,
    }
    return float(total), components
```
