```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号恢复 ----
    # 货箱到 dock 的偏移（已归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist_now = (dx * dx + dy * dy) ** 0.5
    dist_next = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速度（世界系，恢复为 m/s）
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向与 dock 对齐（假设 dock 朝向为世界 x 轴，误差角 = atan2(sin, cos)）
    ch = obs[10]
    sh = obs[11]
    # 对齐度：cos(误差) 的绝对值，1 表示完全对齐（0 或 180 度）
    align = (ch * ch) ** 0.5
    if align > 1.0:
        align = 1.0

    # 车-箱接触
    contact = obs[14]

    # 静态障碍接近度
    sf = obs[15]
    sl = obs[16]
    sr = obs[17]
    obs_near = sf if sf > sl else sl
    if sr > obs_near:
        obs_near = sr

    # ---- 组件 1：货箱向 dock 的进度（delta 距离，主信号）----
    progress = dist_now - dist_next
    crate_to_dock_progress = 6.0 * progress

    # ---- 组件 2：货箱朝向对齐（仅在靠近 dock 时激活，gated）----
    # 门控：距离越近，对齐信号越强
    near_gate = 1.0 / (1.0 + 4.0 * dist_now)
    crate_dock_alignment = 1.5 * near_gate * (align - 0.5)

    # ---- 组件 3：货箱在 dock 附近低速（门控式，不惩罚推进）----
    # 只在货箱已经接近 dock 时，奖励低速；远离时该组件≈0，不影响推进
    settle_gate = 1.0 / (1.0 + 6.0 * dist_now)
    # 低速奖励：速度越小越高，用 bounded 形式
    low_speed_bonus = 1.0 / (1.0 + 8.0 * crate_speed)
    crate_settling = 1.0 * settle_gate * low_speed_bonus

    # ---- 组件 4：轻柔接触（仅在接触时对相对速度做 hinge 惩罚）----
    # 相对速度近似：小车前向速度与货箱速度之差
    cart_v = obs[4] * 3.0
    rel_speed = cart_v - (cvx * obs[2] + cvy * obs[3])
    if rel_speed < 0.0:
        rel_speed = -rel_speed
    if contact > 0.5:
        gentle = -0.8 * max(0.0, rel_speed - 1.0)
    else:
        gentle = 0.0
    gentle_contact = gentle

    # ---- 组件 5：边界规避（hinge，仅在接近边界时生效）----
    cart_x = obs[0]
    cart_y = obs[1]
    bx = 0.0
    if cart_x > 0.8:
        bx = cart_x - 0.8
    elif cart_x < -0.8:
        bx = -0.8 - cart_x
    by = 0.0
    if cart_y > 0.8:
        by = cart_y - 0.8
    elif cart_y < -0.8:
        by = -0.8 - cart_y
    boundary_avoidance = -3.0 * (bx * bx + by * by)

    # ---- 组件 6：避墙（静态障碍接近度 hinge）----
    wall_pen = max(0.0, obs_near - 0.7)
    obstacle_avoidance = -1.0 * wall_pen * wall_pen

    # ---- 组件 7：动作平滑（轻量，避免抖动）----
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_dock_alignment": crate_dock_alignment,
        "crate_settling": crate_settling,
        "gentle_contact": gentle_contact,
        "boundary_avoidance": boundary_avoidance,
        "obstacle_avoidance": obstacle_avoidance,
        "action_smoothness": action_smoothness,
    }

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]

    return (float(total_reward), components)
```