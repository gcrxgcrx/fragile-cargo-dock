```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测中提取可用信号 ----
    # 货箱到泊位的有符号偏移（归一化）
    dx = next_obs[12]
    dy = next_obs[13]
    dx_prev = obs[12]
    dy_prev = obs[13]

    # 货箱到泊位距离（归一化欧氏距离）
    dist = (dx * dx + dy * dy) ** 0.5
    dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5

    # 货箱朝向误差（弧度，0 表示对齐）
    crate_heading = 0.0
    ch = next_obs[10]
    sh = next_obs[11]
    # atan2 的近似：用 sin/cos 构造对齐度，避免 import math
    # 对齐度 = cos(朝向误差)，货箱朝向与泊位朝向(假设为0)对齐
    align = ch  # cos 分量即为朝向对齐度（泊位朝向为0）

    # 货箱速度（归一化）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 接触标志
    contact = next_obs[14]

    # 小车位置
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # 障碍接近度
    obs_front = next_obs[15]
    obs_left = next_obs[16]
    obs_right = next_obs[17]

    # ---- 组件 1：货箱向泊位推进（improvement_delta 主信号）----
    # 用距离减少量作为主进度信号，避免悬停陷阱
    dock_progress = dist_prev - dist
    # 放大到合适量级
    dock_progress = dock_progress * 10.0

    # ---- 组件 2：货箱泊位接近度（bounded_signal，仅作温和的接近引导）----
    # 平滑压缩，避免悬停但提供梯度
    near_dock = 1.0 / (1.0 + 5.0 * dist)
    # 仅在货箱接近泊位时激活（dist < 0.15）
    if dist < 0.15:
        dock_proximity = near_dock * 0.5
    else:
        dock_proximity = 0.0

    # ---- 组件 3：货箱朝向对齐 shaping（仅在接近泊位时激活）----
    # 对齐度从 -1 到 1，映射到 0 到 1
    align_factor = (align + 1.0) * 0.5
    if dist < 0.15:
        heading_align = align_factor * 0.4
    else:
        heading_align = 0.0

    # ---- 组件 4：货箱静止停靠 shaping（仅在接近泊位时激活）----
    # 速度越低越好，用 hinge 形式
    if dist < 0.15:
        speed_penalty = -2.0 * (crate_speed ** 2)
        # 静止奖励
        if crate_speed < 0.05:
            speed_penalty += 0.3
    else:
        speed_penalty = 0.0

    # ---- 组件 5：联合完成代理（joint_condition_proxy）----
    # 位置因子：完全进入泊位
    pos_factor = 0.0
    if abs(dx) <= 0.024 and abs(dy) <= 0.030:
        pos_factor = 1.0
    else:
        # 连续化
        pos_err = abs(dx) / 0.024 + abs(dy) / 0.030
        pos_factor = max(0.0, 1.0 - pos_err * 0.5)

    # 朝向因子
    heading_factor = max(0.0, (align - 0.866) / 0.134)  # cos(30°)=0.866
    heading_factor = min(1.0, heading_factor)

    # 速度因子
    speed_factor = max(0.0, 1.0 - crate_speed / 0.05)
    speed_factor = min(1.0, speed_factor)

    # 联合代理：几何平均，避免塌缩
    joint = (pos_factor * heading_factor * speed_factor + 1e-6) ** (1.0 / 3.0)
    # 仅在货箱接近泊位时激活
    if dist < 0.15:
        joint_condition = joint * 1.5
    else:
        joint_condition = 0.0

    # ---- 组件 6：轻柔接触惩罚（soft_contact_penalty）----
    # 当接触且货箱速度突变大时惩罚（间接推断硬碰撞）
    soft_contact = 0.0
    if contact > 0.5:
        # 货箱速度过快时接触可能造成硬碰撞
        if crate_speed > 0.15:
            soft_contact = -0.5 * (crate_speed - 0.15) ** 2
    # 前向障碍接近度惩罚
    if obs_front > 0.8:
        soft_contact -= 0.1 * (obs_front - 0.8)

    # ---- 组件 7：越界惩罚（out_of_bounds_penalty）----
    # 小车位置边界（归一化后约 [-1, 1]）
    out_of_bounds = 0.0
    if abs(cart_x) > 0.9:
        out_of_bounds -= 0.3 * (abs(cart_x) - 0.9) ** 2
    if abs(cart_y) > 0.9:
        out_of_bounds -= 0.3 * (abs(cart_y) - 0.9) ** 2
    # 货箱越界（通过 dx, dy 间接判断，货箱远离泊位可能越界）
    # 货箱到泊位偏移过大时警告
    if dist > 1.5:
        out_of_bounds -= 0.2 * (dist - 1.5) ** 2

    # ---- 组件 8：动作平滑（action_smoothness，轻量）----
    action_smooth = -0.02 * (action[0] ** 2 + action[1] ** 2)

    # ---- 总奖励 ----
    components = {
        "dock_progress": dock_progress,
        "dock_proximity": dock_proximity,
        "heading_align": heading_align,
        "speed_stop": speed_penalty,
        "joint_condition": joint_condition,
        "soft_contact": soft_contact,
        "out_of_bounds": out_of_bounds,
        "action_smooth": action_smooth,
    }

    total_reward = (
        dock_progress
        + dock_proximity
        + heading_align
        + speed_penalty
        + joint_condition
        + soft_contact
        + out_of_bounds
        + action_smooth
    )

    return (float(total_reward), components)
```