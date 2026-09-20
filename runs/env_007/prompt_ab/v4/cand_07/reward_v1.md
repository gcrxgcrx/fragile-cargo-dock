```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 几何量（由 obs 恢复） ----------
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 朝向误差
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    ang_err = abs(cc)  # cos(heading) 接近 1 表示对齐；用 1-cos 作为误差
    align = (1.0 - cc) if cc < 1.0 else 0.0  # 0 表示完全对齐

    # 货箱速度
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 完成条件（显式从 obs 推断）
    in_tol = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = (cc >= 0.8660254)  # cos(30°) = 0.8660254
    slow = (crate_speed < 0.05)
    done_state = in_tol and aligned and slow

    # ---------- 连续计数 ----------
    if done_state:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---------- 过程型信号（增量形式） ----------
    # 货箱到泊位距离的增量（只在更接近时给分）
    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5
    delta_progress = old_dist - dist
    if delta_progress < 0.0:
        delta_progress = 0.0
    progress_reward = 2.0 * delta_progress

    # 朝向对齐增量（只在变好时给分，且门控在接近泊位时）
    old_cc = float(obs[10])
    delta_align = cc - old_cc
    if delta_align < 0.0:
        delta_align = 0.0
    near_gate = 1.0 if dist < 0.25 else 0.0
    align_reward = 0.5 * near_gate * delta_align

    # ---------- 轻柔度（接触时接近速度惩罚） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---------- 越界惩罚（hinge，仅边界附近） ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    oob = 0.0
    if abs(cx) > 0.85:
        oob += (abs(cx) - 0.85)
    if abs(cy) > 0.85:
        oob += (abs(cy) - 0.85)
    boundary_penalty = -2.0 * oob

    # ---------- 完成状态门控：完成状态下除事件外全部为 0 ----------
    if done_state:
        progress_reward = 0.0
        align_reward = 0.0
        gentleness = 0.0
        boundary_penalty = 0.0

    # ---------- 一次性事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    enter_event = 0.0
    if in_tol and aligned and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    components = {
        "progress": progress_reward,
        "align": align_reward,
        "gentleness": gentleness,
        "boundary": boundary_penalty,
        "enter_event": enter_event,
        "success_event": success_event,
    }
    total = (
        progress_reward
        + align_reward
        + gentleness
        + boundary_penalty
        + enter_event
        + success_event
    )
    return (float(total), components)
```