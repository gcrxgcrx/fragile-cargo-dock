# Response Record

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 泊位几何 ----------
    # 半宽 5.0 m, 半高 4.0 m (由 obs[12]/obs[13] 归一化反推)
    dx = next_obs[12] * 5.0
    dy = next_obs[13] * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    # 完成容差 (环境事实给定): |obs[12]|<=0.024, |obs[13]|<=0.030
    inside = 1.0 if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030) else 0.0

    # ---------- 主信号 1: 货箱到泊位的增量进展 ----------
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    progress = _PREV_DIST[0] - dist          # 本帧更接近泊位为正
    _PREV_DIST[0] = dist
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5
    crate_progress = 2.0 * progress

    # ---------- 主信号 2: 完成事件 (一次性) ----------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    heading_err = next_obs[11]  # sin(angle) 近似角度误差(小角)
    # 更稳的朝向误差: 用 cos 判断是否 > 30° (cos30 ≈ 0.866)
    aligned = 1.0 if next_obs[10] >= 0.866 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0

    done_cond = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0
    if done_cond > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位的一次性奖励
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 5.0

    # ---------- 轻柔度 (接触时惩罚接近速度) ----------
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 接近泊位时的速度抑制 (门控: 仅在接近时激活) ----------
    near_gate = 0.0
    if dist < 1.0:
        near_gate = 1.0 - dist / 1.0
    speed_penalty = -0.3 * near_gate * (crate_speed ** 2)

    # ---------- 越界 hinge 惩罚 ----------
    # 小车位置 obs[0] 归一化到 [-2,2] 裁剪, 但真实范围约 [-1,1]
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    ob_penalty = 0.0
    if cart_x > 0.95:
        ob_penalty -= 0.5 * (cart_x - 0.95)
    if cart_x < -0.95:
        ob_penalty -= 0.5 * (-0.95 - cart_x)
    if cart_y > 0.95:
        ob_penalty -= 0.5 * (cart_y - 0.95)
    if cart_y < -0.95:
        ob_penalty -= 0.5 * (-0.95 - cart_y)

    # ---------- 动作平滑 (轻量) ----------
    smooth = -0.02 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_progress": crate_progress,
        "success_event": success_event,
        "enter_event": enter_event,
        "gentleness": gentleness,
        "speed_penalty_near_dock": speed_penalty,
        "out_of_bounds_penalty": ob_penalty,
        "action_smoothness": smooth,
    }

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```
