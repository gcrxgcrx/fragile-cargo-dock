# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 回合边界检测（obs[18] 单调递增，重置时回落）----
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- 泊位几何（容差取自环境事实）----
    # |obs[12]| <= 0.024, |obs[13]| <= 0.030 视为完全进入
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5
    pdx = float(obs[12])
    pdy = float(obs[13])
    prev_dist = (pdx * pdx + pdy * pdy) ** 0.5

    # ---- 货箱速度（世界系，m/s）----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 朝向误差 ----
    crate_ang = 0.0
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    crate_ang = (sh * sh + ch * ch) ** 0.5
    # 归一化朝向向量
    if crate_ang > 1e-6:
        ch = ch / crate_ang
        sh = sh / crate_ang
    # 泊位朝向假设与小车初始朝向一致（世界系 x 轴）：对齐 = cos 误差
    align = ch  # cos(朝向误差)，范围 [-1,1]；误差 <30° => align > 0.866
    if align < 0.0:
        align = 0.0

    # ---- 组件 1：货箱向泊位的增量推进（只在更接近时给分）----
    progress = prev_dist - dist
    if progress < 0.0:
        progress = 0.0
    r_progress = 12.0 * progress

    # ---- 组件 2：货箱静止度（仅在接近泊位时激活，门控，不做全局收分）----
    near_gate = 0.0
    if dist < 0.15:
        near_gate = 1.0 - dist / 0.15
        if near_gate < 0.0:
            near_gate = 0.0
    still = 1.0 / (1.0 + 20.0 * crate_speed)
    r_still = 0.6 * near_gate * still

    # ---- 组件 3：朝向对齐（接近泊位时激活）----
    r_align = 0.5 * near_gate * align

    # ---- 组件 4：接触轻柔度（唯一教减速的信号）----
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    r_gentle = -0.05 * contact * closing

    # ---- 组件 5：越界 hinge 惩罚（轻量）----
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    ax = abs(cart_x)
    ay = abs(cart_y)
    if ax > 0.92:
        oob += (ax - 0.92)
    if ay > 0.92:
        oob += (ay - 0.92)
    r_oob = -2.0 * oob

    # ---- 组件 6：动作平滑（轻量，避免抖动）----
    a0 = float(action[0])
    a1 = float(action[1])
    r_smooth = -0.01 * (a0 * a0 + a1 * a1)

    # ---- 完成事件判定（连续 10 步）----
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    aligned = 1.0 if align > 0.866 else 0.0
    if inside > 0.5 and slow > 0.5 and aligned > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # ---- 一次性首次进入泊位奖励 ----
    enter_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 20.0

    total = (r_progress + r_still + r_align + r_gentle
             + r_oob + r_smooth + success_event + enter_event)

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_still_near_dock": r_still,
        "crate_align_near_dock": r_align,
        "soft_contact_gentleness": r_gentle,
        "out_of_bounds_penalty": r_oob,
        "action_smoothness": r_smooth,
        "first_enter_dock_event": enter_event,
        "success_event": success_event,
    }

    return float(total), components
```
