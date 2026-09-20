# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测中提取可用信号 ----
    # 货箱到泊位的有符号偏移（归一化）
    dx = next_obs[12]
    dy = next_obs[13]
    dx_old = obs[12]
    dy_old = obs[13]

    # 货箱到泊位的归一化距离
    dist = (dx * dx + dy * dy) ** 0.5
    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5

    # 货箱速度（归一化）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差
    crate_heading = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_heading < 1e-6:
        heading_err = 0.0
    else:
        # 泊位朝向为 0 度（cos=1, sin=0），误差为货箱朝向角
        ch = next_obs[10] / crate_heading
        sh = next_obs[11] / crate_heading
        if ch > 1.0:
            ch = 1.0
        if ch < -1.0:
            ch = -1.0
        # 朝向误差的近似：|sin(theta)| 作为角度偏差的平滑度量
        heading_err = sh if sh >= 0.0 else -sh

    # 接触标志
    contact = next_obs[14]

    # 小车位置（用于越界判断）
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # 障碍接近度
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    # ---- 组件 1: 货箱到泊位的进展（主信号，delta 形式） ----
    # 使用 delta(distance)：靠近泊位为正，远离为负
    progress = (dist_old - dist) * 8.0

    # ---- 组件 2: 泊位内位置质量（接近泊位后激活） ----
    # 门控：货箱接近泊位时才激活
    near_gate = 0.0
    if dist < 0.15:
        near_gate = (0.15 - dist) / 0.15
    # 位置质量：越接近泊位中心越好（有界）
    pos_quality = near_gate * (1.0 / (1.0 + 20.0 * dist))

    # ---- 组件 3: 朝向对齐（接近泊位后激活） ----
    # 门控：货箱接近泊位时才激活
    align_quality = 0.0
    if dist < 0.20:
        align_gate = (0.20 - dist) / 0.20
        # 朝向误差越小越好，30度约 sin(30)=0.5
        align_factor = 1.0 - min(1.0, heading_err / 0.5)
        align_quality = align_gate * align_factor

    # ---- 组件 4: 泊位内低速（接近泊位后激活，门控形式） ----
    # 只在接近泊位时抑制速度，且乘在进展信号上而非全局持续惩罚
    dock_speed_gate = 0.0
    if dist < 0.12:
        dock_speed_gate = (0.12 - dist) / 0.12
    # 速度越低越好，但必须是门控形式
    speed_factor = 1.0 / (1.0 + 30.0 * crate_speed)
    dock_speed_quality = dock_speed_gate * speed_factor

    # ---- 组件 5: 联合完成近似（位置 + 朝向 + 静止） ----
    # 使用几何平均，避免乘积塌缩
    f_pos = 1.0 / (1.0 + 25.0 * dist)
    f_align = 1.0 - min(1.0, heading_err / 0.6)
    f_speed = 1.0 / (1.0 + 25.0 * crate_speed)
    joint = (f_pos * f_align * f_speed) ** (1.0 / 3.0)
    # 只在接近泊位时激活联合信号
    joint_gate = 0.0
    if dist < 0.15:
        joint_gate = (0.15 - dist) / 0.15
    joint_quality = joint_gate * joint

    # ---- 组件 6: 软接触惩罚（仅在接触且速度大时激活） ----
    # 推断：接触 + 货箱速度大 => 可能硬碰撞
    soft_contact_penalty = 0.0
    if contact > 0.5:
        # 货箱速度超过安全阈值时惩罚
        speed_excess = crate_speed - 0.3
        if speed_excess > 0.0:
            soft_contact_penalty = -0.5 * speed_excess

    # ---- 组件 7: 越界惩罚（hinge 形式） ----
    # 小车位置边界（归一化坐标，边界约为 1.0）
    out_penalty = 0.0
    cart_bound = 0.95
    if cart_x > cart_bound:
        out_penalty -= 2.0 * (cart_x - cart_bound)
    if cart_x < -cart_bound:
        out_penalty -= 2.0 * (-cart_x - cart_bound)
    if cart_y > cart_bound:
        out_penalty -= 2.0 * (cart_y - cart_bound)
    if cart_y < -cart_bound:
        out_penalty -= 2.0 * (-cart_y - cart_bound)

    # ---- 组件 8: 动作平滑（轻量） ----
    action_smoothness = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---- 组件 9: 障碍接近惩罚（hinge） ----
    obstacle_penalty = 0.0
    if sensor_front > 0.7:
        obstacle_penalty -= 0.3 * (sensor_front - 0.7)
    if sensor_left > 0.7:
        obstacle_penalty -= 0.3 * (sensor_left - 0.7)
    if sensor_right > 0.7:
        obstacle_penalty -= 0.3 * (sensor_right - 0.7)

    # ---- 总奖励 ----
    components = {}
    components["crate_to_dock_progress"] = progress
    components["dock_position_quality"] = 3.0 * pos_quality
    components["dock_alignment_quality"] = 2.0 * align_quality
    components["dock_speed_quality"] = 2.0 * dock_speed_quality
    components["joint_completion"] = 4.0 * joint_quality
    components["soft_contact_penalty"] = soft_contact_penalty
    components["out_of_bounds_penalty"] = out_penalty
    components["action_smoothness"] = action_smoothness
    components["obstacle_penalty"] = obstacle_penalty

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
