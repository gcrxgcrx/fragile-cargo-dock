```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测中提取信号 ----
    # 货箱到泊位的有符号偏移（归一化）
    crate_dock_x = next_obs[12]
    crate_dock_y = next_obs[13]
    prev_dock_x = obs[12]
    prev_dock_y = obs[13]

    # 货箱世界系速度（归一化，m/s / 3.0）
    crate_vx = next_obs[8]
    crate_vy = next_obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向误差（弧度）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_error = (crate_sin * crate_sin + (1.0 - crate_cos) * (1.0 - crate_cos)) ** 0.5

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    # ---- 1. 货箱向泊位靠近的进度信号（delta distance）----
    prev_dist = (prev_dock_x * prev_dock_x + prev_dock_y * prev_dock_y) ** 0.5
    curr_dist = (crate_dock_x * crate_dock_x + crate_dock_y * crate_dock_y) ** 0.5
    progress = prev_dist - curr_dist
    crate_to_dock_progress = 2.0 * progress

    # ---- 2. 货箱停靠质量（联合条件软代理）----
    # 位置因子：完全进入泊位 |x|<=0.024, |y|<=0.030
    pos_x_factor = max(0.0, 1.0 - abs(crate_dock_x) / 0.024)
    pos_y_factor = max(0.0, 1.0 - abs(crate_dock_y) / 0.030)
    pos_factor = pos_x_factor * pos_y_factor

    # 朝向因子：误差 < 30° (0.5236 rad)
    heading_factor = max(0.0, 1.0 - heading_error / 0.5236)

    # 静止因子：速度 < 0.05 m/s 对应归一化 0.05/3.0 ≈ 0.01667
    speed_factor = max(0.0, 1.0 - crate_speed / 0.01667)

    # 几何平均，避免乘积塌缩
    docking_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 3.0 * docking_quality

    # ---- 3. 接近泊位时的速度抑制 ----
    # 仅在货箱接近泊位时启用（距离归一化阈值 0.15）
    near_dock = max(0.0, 1.0 - curr_dist / 0.15)
    crate_speed_penalty_near_dock = -1.0 * near_dock * crate_speed

    # ---- 4. 软接触惩罚（间接推断硬碰撞）----
    # 接触时货箱速度过大视为硬碰撞风险
    if contact > 0.5:
        excess_speed = max(0.0, crate_speed - 0.02)
        soft_contact_penalty = -1.5 * excess_speed
    else:
        soft_contact_penalty = 0.0

    # ---- 5. 越界惩罚（hinge，边界附近生效）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 小车位置归一化，接近 ±1 时惩罚
    cart_edge = max(abs(cart_x), abs(cart_y))
    cart_bound_penalty = -2.0 * max(0.0, cart_edge - 0.85)

    # 货箱位置恢复（近似）：用货箱到泊位偏移 + 泊位位置估计
    # 货箱越界通过 crate_dock 偏移的极端值间接检测
    crate_edge = max(abs(crate_dock_x), abs(crate_dock_y))
    crate_bound_penalty = -2.0 * max(0.0, crate_edge - 0.9)

    out_of_bounds_penalty = cart_bound_penalty + crate_bound_penalty

    # ---- 6. 障碍接近惩罚（防止撞墙）----
    obstacle_penalty = -0.5 * (max(0.0, sensor_front - 0.8) +
                                max(0.0, sensor_left - 0.8) +
                                max(0.0, sensor_right - 0.8))

    # ---- 7. 动作平滑（轻量，避免抖动）----
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_docking_quality": float(crate_docking_quality),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = sum(components.values())
    return (float(total_reward), components)
```