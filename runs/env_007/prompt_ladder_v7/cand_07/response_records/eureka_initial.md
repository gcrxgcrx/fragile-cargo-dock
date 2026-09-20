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

    # ---------- 基础量 ----------
    dx = float(next_obs[12])   # 货箱到泊位 x 偏移 / 半宽
    dy = float(next_obs[13])   # 货箱到泊位 y 偏移 / 半高
    dx0 = float(obs[12])
    dy0 = float(obs[13])

    # 归一化距离（用泊位容差尺度归一，方便比较）
    dist = (dx * dx + dy * dy) ** 0.5
    dist0 = (dx0 * dx0 + dy0 * dy0) ** 0.5

    # 朝向误差（弧度）
    cos_h = float(next_obs[10])
    sin_h = float(next_obs[11])
    ang_err = (sin_h * sin_h) ** 0.5          # |sin(theta_err)| 近似，越小越对齐
    aligned = 1.0 if ang_err < 0.5 else 0.0   # <30°

    # 货箱速度
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 完成谓词（严格用环境给定容差）
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    done_now = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if done_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 组件 1：推进增量（主信号） ----------
    # 只奖励"这一帧更接近泊位"，停在附近不再收分
    progress = dist0 - dist
    if progress < 0.0:
        progress = 0.0
    # 对齐门控：朝向越差，推进收益越低（但不为负）
    align_gate = 1.0 - 0.5 * min(1.0, ang_err)
    push_reward = 60.0 * progress * align_gate

    # ---------- 组件 2：轻柔度（接触 + 接近速度惩罚） ----------
    crate_along = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -6.0 * contact * closing   # 1 m/s 接近 => -6，与推进同量级

    # ---------- 组件 3：接近泊位时的高速惩罚（hinge，非全局） ----------
    near_gate = 1.0 if dist < 0.25 else 0.0
    speed_pen = -8.0 * near_gate * max(0.0, crate_speed - 0.05)

    # ---------- 组件 4：越界守卫（小车 + 货箱） ----------
    cx = abs(float(next_obs[0]))
    cy = abs(float(next_obs[1]))
    cart_over = max(0.0, cx - 0.90) + max(0.0, cy - 0.90)
    # 货箱世界坐标近似（用泊位偏移 + 已知泊位位置不可得，改用相对量保守估计）
    crate_over = max(0.0, abs(dx) - 0.9) + max(0.0, abs(dy) - 0.9)
    bounds_pen = -80.0 * (cart_over + crate_over)

    # ---------- 组件 5：停稳期每步收益（仅未完成时给） ----------
    # 处于"泊位内 + 对齐 + 慢"但尚未走完 10 步 => 每步正收益
    if done_now > 0.5 and not _PAID[0]:
        settled_bonus = 20.0
    else:
        settled_bonus = 0.0

    # ---------- 组件 6：首次进入泊位（一次性） ----------
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 15.0

    # ---------- 组件 7：一次性完成事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
        settled_bonus = 0.0   # 完成当步不再叠加停稳收益

    # ---------- 完成态下其余组件必须为 0 ----------
    if done_now > 0.5 and _PAID[0]:
        push_reward = 0.0
        gentleness = 0.0
        speed_pen = 0.0
        settled_bonus = 0.0
        enter_bonus = 0.0

    total = (push_reward + gentleness + speed_pen + bounds_pen
             + settled_bonus + enter_bonus + success_event)

    components = {
        "push_reward": float(push_reward),
        "gentleness": float(gentleness),
        "speed_pen": float(speed_pen),
        "bounds_pen": float(bounds_pen),
        "settled_bonus": float(settled_bonus),
        "enter_bonus": float(enter_bonus),
        "success_event": float(success_event),
    }
    return float(total), components
```
