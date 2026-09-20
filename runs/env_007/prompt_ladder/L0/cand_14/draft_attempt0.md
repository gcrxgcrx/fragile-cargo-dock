```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 观测解包（仅使用已声明维度）----
    cart_x = obs[0]
    cart_y = obs[1]
    cart_cos = obs[2]
    cart_sin = obs[3]
    cart_v = obs[4]
    cart_yaw = obs[5]

    crate_rel_x = obs[6]
    crate_rel_y = obs[7]
    crate_vx = obs[8]
    crate_vy = obs[9]
    crate_cos = obs[10]
    crate_sin = obs[11]

    dock_x = obs[12]
    dock_y = obs[13]
    contact = obs[14]

    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    time_frac = obs[18]

    n_dock_x = next_obs[12]
    n_dock_y = next_obs[13]
    n_crate_vx = next_obs[8]
    n_crate_vy = next_obs[9]

    drive = action[0]
    steer = action[1]

    # ---- 货箱到泊位距离（归一化单位）----
    dist = (dock_x * dock_x + dock_y * dock_y) ** 0.5
    n_dist = (n_dock_x * n_dock_x + n_dock_y * n_dock_y) ** 0.5

    # ---- 1) 主进度：货箱向泊位靠近的 delta ----
    progress = (dist - n_dist) * 6.0
    # 限制单步幅度，避免异常跳变
    if progress > 1.0:
        progress = 1.0
    if progress < -1.0:
        progress = -1.0

    # ---- 2) 停靠质量：位置 + 朝向 + 速度 的联合连续 proxy ----
    # 位置因子：越接近泊位中心越接近 1
    pos_err = dist
    pos_factor = 1.0 / (1.0 + 12.0 * pos_err)

    # 朝向因子：货箱朝向误差（相对对齐，cos 越接近 1 越好）
    # 用 |sin| 近似角度偏离（0 表示对齐）
    align = crate_cos  # 期望与泊位对齐时 cos ~ 1
    if align < 0.0:
        align = 0.0
    orient_factor = 1.0 / (1.0 + 4.0 * (1.0 - align))

    # 速度因子：货箱速度越低越接近 1
    crate_speed = (n_crate_vx * n_crate_vx + n_crate_vy * n_crate_vy) ** 0.5
    speed_factor = 1.0 / (1.0 + 15.0 * crate_speed)

    # 几何平均，避免乘积塌缩
    joint = (pos_factor * orient_factor * speed_factor) ** (1.0 / 3.0)
    dock_quality = 3.0 * joint

    # ---- 3) 接近泊位时的速度抑制（hinge：仅在接近时启用）----
    near_gate = 1.0 / (1.0 + 8.0 * pos_err)  # 距离越近越接近 1
    speed_over = crate_speed - 0.05
    if speed_over < 0.0:
        speed_over = 0.0
    speed_penalty = -2.0 * near_gate * speed_over

    # ---- 4) 软接触/碰撞抑制（间接推断：接触时货箱速度突变过大则惩罚）----
    contact_penalty = 0.0
    if contact > 0.5:
        # 接触时货箱速度过高视为硬推
        hard_push = crate_speed - 0.6
        if hard_push < 0.0:
            hard_push = 0.0
        contact_penalty = -1.5 * hard_push

    # ---- 5) 越界惩罚（hinge：接近仓库边界时）----
    # 小车位置：|x|,|y| 接近 1 为边界
    cart_edge = abs(cart_x) - 0.85
    if cart_edge < 0.0:
        cart_edge = 0.0
    cart_edge_y = abs(cart_y) - 0.85
    if cart_edge_y < 0.0:
        cart_edge_y = 0.0
    # 货箱位置由相对量粗略估计：|dock 偏移| 大表示远离泊位
    crate_edge = abs(dock_x) - 0.9
    if crate_edge < 0.0:
        crate_edge = 0.0
    crate_edge_y = abs(dock_y) - 0.9
    if crate_edge_y < 0.0:
        crate_edge_y = 0.0
    oob_penalty = -3.0 * (cart_edge + cart_edge_y + crate_edge + crate_edge_y)

    # ---- 6) 障碍接近惩罚（hinge：传感器接近接触时）----
    obs_penalty = 0.0
    if sensor_front > 0.85:
        obs_penalty += (sensor_front - 0.85)
    if sensor_left > 0.9:
        obs_penalty += (sensor_left - 0.9)
    if sensor_right > 0.9:
        obs_penalty += (sensor_right - 0.9)
    obstacle_penalty = -1.0 * obs_penalty

    # ---- 7) 动作平滑（轻量，防止抖动）----
    smooth_penalty = -0.05 * (drive * drive + steer * steer)

    # ---- 总奖励 ----
    total = (
        progress
        + dock_quality
        + speed_penalty
        + contact_penalty
        + oob_penalty
        + obstacle_penalty
        + smooth_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress),
        "crate_docking_quality": float(dock_quality),
        "crate_speed_penalty_near_dock": float(speed_penalty),
        "soft_contact_penalty": float(contact_penalty),
        "out_of_bounds_penalty": float(oob_penalty),
        "obstacle_proximity_penalty": float(obstacle_penalty),
        "action_smoothness": float(smooth_penalty),
    }

    return float(total), components
```