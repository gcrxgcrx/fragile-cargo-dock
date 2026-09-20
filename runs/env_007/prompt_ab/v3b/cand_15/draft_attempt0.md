```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测（基于 obs[18] 单调递增、重置回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 泊位几何（仅使用环境事实给出的容差） ----------
    # |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 视为"完全进入泊位"
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    adx = dx if dx >= 0.0 else -dx
    ady = dy if dy >= 0.0 else -dy

    inside_x = 1.0 if adx <= 0.024 else 0.0
    inside_y = 1.0 if ady <= 0.030 else 0.0
    inside = 1.0 if (inside_x > 0.5 and inside_y > 0.5) else 0.0

    # 归一化距离（用容差尺度做归一，便于比较）
    dist_norm = (adx / 0.024) + (ady / 0.030)

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 货箱朝向误差 ----------
    ccos = float(next_obs[10])
    csin = float(next_obs[11])
    # 与泊位对齐：假设泊位朝向与货箱期望朝向一致，用 |cos| 与 |sin| 组合衡量
    # 朝向误差角 theta 满足 cos(theta) = ccos（对齐时 ccos≈1）
    align_err = 1.0 - ccos  # 0 表示对齐，2 表示反向
    aligned = 1.0 if align_err <= 0.134 else 0.0  # cos(30°)=0.866 -> err<=0.134

    # ---------- 组件 1：货箱向泊位的增量推进（主进度信号，增量形式） ----------
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist_norm
    delta_dist = _PREV_DIST[0] - dist_norm  # 正 = 更接近
    _PREV_DIST[0] = dist_norm

    progress = 2.0 * delta_dist
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5

    # ---------- 组件 2：进入泊位的首次一次性奖励 ----------
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 20.0

    # ---------- 组件 3：接近泊位且对齐时的低速门控塑形（仅接近时激活） ----------
    # 仅在已经比较接近泊位时启用，避免全局持续收分
    near_gate = 0.0
    if dist_norm < 4.0:
        near_gate = 1.0 - dist_norm / 4.0
        if near_gate < 0.0:
            near_gate = 0.0
    # 低速因子：速度越低越好，但只在接近泊位时作为门控
    speed_factor = 1.0 / (1.0 + 8.0 * crate_speed)
    align_factor = 1.0 / (1.0 + 6.0 * align_err)
    dock_quality = 3.0 * near_gate * speed_factor * align_factor

    # ---------- 组件 4：接触轻柔度（唯一教会减速的信号） ----------
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 组件 5：越界 hinge 惩罚（小车与货箱） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    out_pen = 0.0
    if cart_x > 0.95:
        out_pen -= 1.0 * (cart_x - 0.95)
    if cart_x < -0.95:
        out_pen -= 1.0 * (-0.95 - cart_x)
    if cart_y > 0.95:
        out_pen -= 1.0 * (cart_y - 0.95)
    if cart_y < -0.95:
        out_pen -= 1.0 * (-0.95 - cart_y)

    # ---------- 组件 6：完成事件（一次性，连续 10 步满足） ----------
    if inside > 0.5 and aligned > 0.5 and crate_speed < 0.05:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    components = {
        "progress": progress,
        "enter_bonus": enter_bonus,
        "dock_quality": dock_quality,
        "gentleness": gentleness,
        "out_of_bounds": out_pen,
        "success_event": success_event,
    }

    total = (
        progress
        + enter_bonus
        + dock_quality
        + gentleness
        + out_pen
        + success_event
    )
    return (float(total), components)
```