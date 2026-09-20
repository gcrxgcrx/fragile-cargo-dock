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

    # ---------- 泊位几何（来自环境事实） ----------
    # 货箱中心到泊位中心有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist_norm = (dx * dx + dy * dy) ** 0.5

    # 完成容差：|dx|<=0.024, |dy|<=0.030
    in_dock = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # 朝向误差
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    # 朝向误差角度（弧度），用 cos 分量近似
    heading_err = abs(crate_sin)  # sin(误差)；误差<30° => |sin|<0.5
    aligned = 1.0 if heading_err < 0.5 else 0.0

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow = 1.0 if crate_speed < 0.05 else 0.0

    # 完成状态
    completed_state = 1.0 if (in_dock > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    # ---------- 连续计数 ----------
    if completed_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ---------- 一次性完成事件 ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 一次性首次进入泊位奖励 ----------
    entered_event = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_event = 30.0
    components["entered_event"] = entered_event

    # ---------- 完成状态下：除一次性事件外其余组件必须为 0 ----------
    if completed_state > 0.5:
        components["crate_progress"] = 0.0
        components["align_progress"] = 0.0
        components["gentleness"] = 0.0
        components["speed_near_dock"] = 0.0
        components["obstacle_penalty"] = 0.0
        total = success_event + entered_event
        return (float(total), components)

    # ---------- 主推进信号：货箱到泊位距离的增量 ----------
    # 用 obs（动作前）与 next_obs（动作后）的归一化距离差
    dx_old = float(obs[12])
    dy_old = float(obs[13])
    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5
    progress = dist_old - dist_norm  # 正 = 更接近
    # 只在接近时给正分；远离时给轻微负分（保持同向）
    crate_progress = 5.0 * progress
    components["crate_progress"] = crate_progress

    # ---------- 朝向对齐增量（门控 × 增量） ----------
    # 用对齐度作为门控，乘在"靠近增量"上，避免持续收分
    align_deg = max(0.0, 1.0 - heading_err / 0.5)  # 0..1
    align_progress = 2.0 * align_deg * max(0.0, progress)
    components["align_progress"] = align_progress

    # ---------- 接触轻柔度 ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing
    components["gentleness"] = gentleness

    # ---------- 接近泊位时的速度抑制（仅惩罚，不持续给分） ----------
    # 距离泊位较近时，货箱速度过快给惩罚
    near_dock = 1.0 if dist_norm < 0.15 else 0.0
    speed_over = max(0.0, crate_speed - 0.15)
    speed_near_dock = -0.3 * near_dock * speed_over
    components["speed_near_dock"] = speed_near_dock

    # ---------- 障碍接近惩罚（hinge） ----------
    sf = float(next_obs[15])
    sl = float(next_obs[16])
    sr = float(next_obs[17])
    obs_pen = 0.0
    if sf > 0.8:
        obs_pen -= 0.2 * (sf - 0.8)
    if sl > 0.8:
        obs_pen -= 0.2 * (sl - 0.8)
    if sr > 0.8:
        obs_pen -= 0.2 * (sr - 0.8)
    components["obstacle_penalty"] = obs_pen

    total = (crate_progress + align_progress + gentleness +
             speed_near_dock + obs_pen + success_event + entered_event)

    return (float(total), components)
```