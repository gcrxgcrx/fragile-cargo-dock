```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# 泊位几何容差（来自环境事实）
_DOCK_X_TOL = 0.024
_DOCK_Y_TOL = 0.030
_ALIGN_TOL = 0.5          # cos(60°) 的保守界，实际要求 <30° 即 cos>0.866
_SPEED_TOL = 0.05 / 3.0   # 归一化速度阈值


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 完成条件（从 obs 显式推断，容差取自环境事实） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    inside = 1.0 if (abs(dx) <= _DOCK_X_TOL and abs(dy) <= _DOCK_Y_TOL) else 0.0

    # 货箱朝向误差：cos 值越大越对齐
    ccos = float(next_obs[10])
    # 归一化朝向向量（防御性）
    cnorm = (ccos * ccos + float(next_obs[11]) * float(next_obs[11])) ** 0.5
    if cnorm > 1e-6:
        ccos = ccos / cnorm
    align_ok = 1.0 if ccos >= 0.866 else 0.0   # 误差 < 30°

    # 货箱速度（归一化）
    cvx = float(next_obs[8])
    cvy = float(next_obs[9])
    cspeed = (cvx * cvx + cvy * cvy) ** 0.5
    slow_ok = 1.0 if cspeed <= _SPEED_TOL else 0.0

    docked_state = 1.0 if (inside > 0.5 and align_ok > 0.5 and slow_ok > 0.5) else 0.0

    if docked_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 一次性完成事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位（一次性）
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    # ---------- 主信号：货箱到泊位的增量推进 ----------
    # 用归一化偏移的欧氏距离作为度量，只在"这一帧更接近"时给分
    old_d = (float(obs[12]) ** 2 + float(obs[13]) ** 2) ** 0.5
    new_d = (dx * dx + dy * dy) ** 0.5
    progress_delta = old_d - new_d
    if progress_delta < 0.0:
        progress_delta = 0.0
    # 门控：朝向对齐度乘在增量上（对齐越好，推进得分越高）
    align_gate = 0.5 + 0.5 * max(0.0, ccos)
    crate_progress = 2.0 * progress_delta * align_gate

    # ---------- 接触轻柔度（唯一能教"接近泊位时减速"的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 接近泊位时的速度抑制（hinge，仅在接近且过快时惩罚） ----------
    near_dock = 1.0 if new_d < 0.25 else 0.0
    speed_excess = cspeed - _SPEED_TOL
    if speed_excess < 0.0:
        speed_excess = 0.0
    dock_speed_penalty = -0.5 * near_dock * (speed_excess ** 2)

    # ---------- 越界防护（hinge，仅在接近边界时轻罚） ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.9:
        oob += (abs(cart_x) - 0.9) ** 2
    if abs(cart_y) > 0.9:
        oob += (abs(cart_y) - 0.9) ** 2
    if new_d > 0.6:
        oob += (new_d - 0.6) ** 2
    oob_penalty = -1.0 * oob

    # ---------- 动作平滑（轻量，可选） ----------
    smooth_penalty = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---------- 完成状态下除一次性事件外必须恰好为 0 ----------
    if docked_state > 0.5:
        crate_progress = 0.0
        gentleness = 0.0
        dock_speed_penalty = 0.0
        oob_penalty = 0.0
        smooth_penalty = 0.0

    components = {
        "crate_progress": crate_progress,
        "gentleness": gentleness,
        "dock_speed_penalty": dock_speed_penalty,
        "out_of_bounds_penalty": oob_penalty,
        "action_smoothness": smooth_penalty,
        "enter_dock_event": enter_event,
        "success_event": success_event,
    }

    total_reward = (
        crate_progress
        + gentleness
        + dock_speed_penalty
        + oob_penalty
        + smooth_penalty
        + enter_event
        + success_event
    )

    return float(total_reward), components
```