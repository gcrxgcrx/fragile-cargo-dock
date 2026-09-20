# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到泊位的有符号偏移（归一化）
    dx = next_obs[12]
    dy = next_obs[13]
    dx_prev = obs[12]
    dy_prev = obs[13]

    # 货箱到泊位距离（归一化空间）
    dist = (dx * dx + dy * dy) ** 0.5
    dist_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5

    # 货箱速度（世界系，归一化）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    heading_err = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if heading_err < 1e-6:
        angle_err = 0.0
    else:
        # atan2 近似：用 sin 分量作为误差代理
        angle_err = abs(next_obs[11])

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    # 小车位置
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # ---------- 1. 货箱向泊位推进（主信号，delta 形式） ----------
    progress = dist_prev - dist
    # 限制单步 delta 的极端值
    if progress > 0.1:
        progress = 0.1
    elif progress < -0.1:
        progress = -0.1
    crate_to_dock_progress = 8.0 * progress

    # ---------- 2. 货箱停靠质量（接近泊位时激活） ----------
    # 接近度因子：距离越近越接近 1
    near_factor = 1.0 / (1.0 + 8.0 * dist)

    # 位置精度：进入泊位区域
    pos_err = (dx * dx + dy * dy) ** 0.5
    pos_factor = 1.0 / (1.0 + 20.0 * pos_err)

    # 朝向对齐因子（误差 < 30° ≈ 0.524 rad）
    angle_factor = 1.0 / (1.0 + 4.0 * angle_err)

    # 静止因子（速度 < 0.05 m/s → 归一化 0.05/3.0 ≈ 0.0167）
    speed_factor = 1.0 / (1.0 + 60.0 * crate_speed)

    # 联合条件（几何平均，避免塌缩）
    docking_quality = (pos_factor * angle_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 3.0 * near_factor * docking_quality

    # ---------- 3. 接近泊位时的速度抑制（防止滑过） ----------
    if dist < 0.15:
        speed_penalty = -1.5 * crate_speed
    else:
        speed_penalty = 0.0

    # ---------- 4. 软接触惩罚（间接推断硬碰撞） ----------
    # 接触时若货箱速度突变大，视为硬碰撞风险
    if contact > 0.5:
        # 货箱速度与小车速度耦合，速度过大时惩罚接触
        speed_mismatch = crate_speed
        if speed_mismatch > 0.1:
            soft_contact = -1.0 * (speed_mismatch - 0.1)
        else:
            soft_contact = 0.0
    else:
        soft_contact = 0.0

    # ---------- 5. 越界惩罚（hinge 形式） ----------
    # 小车位置边界（归一化后约 ±1.0，留出安全余量）
    cart_bound = 0.0
    if abs(cart_x) > 0.85:
        cart_bound += -2.0 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        cart_bound += -2.0 * (abs(cart_y) - 0.85)

    # 货箱到泊位偏移过大（可能越界）
    crate_bound = 0.0
    crate_dist_norm = (dx * dx + dy * dy) ** 0.5
    if crate_dist_norm > 1.5:
        crate_bound = -2.0 * (crate_dist_norm - 1.5)

    out_of_bounds = cart_bound + crate_bound

    # ---------- 6. 障碍接近惩罚（hinge，仅接近时生效） ----------
    obstacle_penalty = 0.0
    if sensor_front > 0.7:
        obstacle_penalty += -0.5 * (sensor_front - 0.7)
    if sensor_left > 0.7:
        obstacle_penalty += -0.5 * (sensor_left - 0.7)
    if sensor_right > 0.7:
        obstacle_penalty += -0.5 * (sensor_right - 0.7)

    # ---------- 7. 动作平滑（轻量） ----------
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 总和 ----------
    components = {}
    components["crate_to_dock_progress"] = crate_to_dock_progress
    components["crate_docking_quality"] = crate_docking_quality
    components["crate_speed_near_dock"] = speed_penalty
    components["soft_contact"] = soft_contact
    components["out_of_bounds"] = out_of_bounds
    components["obstacle_penalty"] = obstacle_penalty
    components["action_smoothness"] = action_smoothness

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + speed_penalty
        + soft_contact
        + out_of_bounds
        + obstacle_penalty
        + action_smoothness
    )

    return (float(total_reward), components)
```
