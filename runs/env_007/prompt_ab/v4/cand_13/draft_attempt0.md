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

    components = {}

    # ---------- 几何量 ----------
    # 泊位半宽/半高（归一化后容差）：|obs[12]|<=0.024, |obs[13]|<=0.030
    dock_x = float(next_obs[12])
    dock_y = float(next_obs[13])
    dock_x_old = float(obs[12])
    dock_y_old = float(obs[13])

    # 归一化距离（用于增量）
    dist_new = (dock_x * dock_x + dock_y * dock_y) ** 0.5
    dist_old = (dock_x_old * dock_x_old + dock_y_old * dock_y_old) ** 0.5

    # 货箱朝向误差 (rad)
    crate_cos = float(next_obs[10])
    crate_sin = float(next_obs[11])
    heading_err = abs((crate_sin * crate_sin) ** 0.5)  # 仅用于构造，下面用 atan2 等价
    # 用 cos 值判断对齐度（cos 越接近 1 越对齐），避免 atan2 不可用
    align_cos = crate_cos  # 货箱朝向余弦，若朝向对齐则接近 1

    # 货箱速度（世界系，归一化）
    cvx = float(next_obs[8])
    cvy = float(next_obs[9])
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5  # 归一化速度，1.0 对应 3 m/s

    # 完成状态判定
    inside = (abs(dock_x) <= 0.024) and (abs(dock_y) <= 0.030)
    aligned = align_cos >= 0.866  # cos(30°)
    slow = crate_speed < 0.05 / 3.0  # 0.05 m/s 归一化
    complete_state = inside and aligned and slow

    # ---------- 1) 主推进：增量式靠近泊位 ----------
    # 仅当货箱更接近泊位时给分（增量形式，避免悬停收割）
    delta_progress = dist_old - dist_new
    if delta_progress > 0.0:
        # 对齐门控：对齐越好，靠近增量收益越大
        align_gate = 0.5 + 0.5 * max(0.0, align_cos)
        crate_to_dock_progress = 3.0 * delta_progress * align_gate
    else:
        crate_to_dock_progress = 0.0

    # ---------- 2) 轻柔接触惩罚（唯一教会减速的信号） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    soft_contact_penalty = -0.05 * contact * closing

    # ---------- 3) 接近泊位时的速度抑制（hinge，仅接近时激活） ----------
    # 距离泊位较近时，若货箱速度过快则惩罚（只在未完成时）
    near_dock = dist_new < 0.15
    speed_penalty_near_dock = 0.0
    if near_dock and not complete_state:
        # 超过 0.15 m/s（归一化 0.05）的部分惩罚
        excess = crate_speed - 0.05
        if excess > 0.0:
            speed_penalty_near_dock = -0.5 * excess * excess

    # ---------- 4) 边界惩罚（hinge，仅接近边界时激活） ----------
    # 小车位置 obs[0], obs[1] 归一化到 [-2,2]，接近 ±1.8 时惩罚
    out_of_bounds_penalty = 0.0
    for pos in (float(next_obs[0]), float(next_obs[1])):
        if abs(pos) > 1.8:
            out_of_bounds_penalty -= 0.5 * (abs(pos) - 1.8) ** 2

    # ---------- 5) 障碍接近惩罚（hinge，仅在很近时） ----------
    obstacle_penalty = 0.0
    for s in (float(next_obs[15]), float(next_obs[16]), float(next_obs[17])):
        if s > 0.9:
            obstacle_penalty -= 0.2 * (s - 0.9) ** 2

    # ---------- 6) 完成状态清零：除一次性事件外其余组件必须恰好为 0 ----------
    if complete_state:
        crate_to_dock_progress = 0.0
        soft_contact_penalty = 0.0
        speed_penalty_near_dock = 0.0
        out_of_bounds_penalty = 0.0
        obstacle_penalty = 0.0

    # ---------- 7) 连续计数与一次性事件 ----------
    if complete_state:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位（一次性）
    entered_event = 0.0
    if inside and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_event = 20.0

    components['crate_to_dock_progress'] = crate_to_dock_progress
    components['soft_contact_penalty'] = soft_contact_penalty
    components['speed_penalty_near_dock'] = speed_penalty_near_dock
    components['out_of_bounds_penalty'] = out_of_bounds_penalty
    components['obstacle_penalty'] = obstacle_penalty
    components['success_event'] = success_event
    components['entered_event'] = entered_event

    total_reward = (
        crate_to_dock_progress
        + soft_contact_penalty
        + speed_penalty_near_dock
        + out_of_bounds_penalty
        + obstacle_penalty
        + success_event
        + entered_event
    )

    return float(total_reward), components
```