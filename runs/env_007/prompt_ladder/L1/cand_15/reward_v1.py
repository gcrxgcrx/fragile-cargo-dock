def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测中恢复任务相关量 ----
    # 货箱到泊位的有符号偏移（归一化）
    dock_x = next_obs[12]
    dock_y = next_obs[13]
    prev_dock_x = obs[12]
    prev_dock_y = obs[13]

    # 货箱到泊位距离（归一化欧氏距离）
    dist = (dock_x * dock_x + dock_y * dock_y) ** 0.5
    prev_dist = (prev_dock_x * prev_dock_x + prev_dock_y * prev_dock_y) ** 0.5

    # 货箱速度（归一化，世界系）
    crate_vx = next_obs[8]
    crate_vy = next_obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向误差（相对泊位对齐，取朝向角的绝对值，归一化到 [0,1]）
    crate_heading = next_obs[11]
    align_err = crate_heading if crate_heading >= 0.0 else -crate_heading

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    front_prox = next_obs[15]
    left_prox = next_obs[16]
    right_prox = next_obs[17]

    # 泊位几何阈值（完全进入）
    in_dock_x = 1.0 if (dock_x <= 0.024 and dock_x >= -0.024) else 0.0
    in_dock_y = 1.0 if (dock_y <= 0.030 and dock_y >= -0.030) else 0.0

    # ---- 1) 主任务进展：货箱向泊位靠近的 delta 信号 ----
    progress = prev_dist - dist
    # 限制单步 delta 极端值，避免刷分
    progress = progress / (1.0 + (progress if progress >= 0.0 else -progress))
    crate_to_dock_progress = 3.0 * progress

    # ---- 2) 泊位接近度（bounded，仅在接近时提供梯度，防止悬停） ----
    proximity = 1.0 / (1.0 + 6.0 * dist)
    crate_dock_proximity = 0.4 * proximity

    # ---- 3) 泊位内对齐质量：位置 + 朝向 + 低速的联合门控 ----
    # 位置因子：越接近泊位中心越高
    pos_factor = 1.0 / (1.0 + 20.0 * dist)
    # 朝向因子：朝向误差越小越高（align_err 归一化约 [0,2]）
    align_factor = 1.0 / (1.0 + 6.0 * align_err)
    # 速度因子：速度越低越高
    speed_factor = 1.0 / (1.0 + 8.0 * crate_speed)
    # 联合几何平均，避免乘积塌缩
    joint = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    # 仅在接近泊位时激活（门控），避免全局持续收分
    near_gate = 1.0 / (1.0 + 10.0 * dist)
    crate_docking_quality = 1.2 * joint * near_gate

    # ---- 4) 完全进入泊位的稀疏奖励（连续化） ----
    inside = in_dock_x * in_dock_y
    crate_inside_bonus = 0.5 * inside

    # ---- 5) 轻柔接触：仅在接触时对货箱速度做 hinge 惩罚 ----
    # 只在接触状态下、且货箱速度超过安全阈值时惩罚
    speed_excess = crate_speed - 0.15
    if speed_excess < 0.0:
        speed_excess = 0.0
    soft_contact_penalty = -0.5 * contact * speed_excess

    # ---- 6) 泊位附近速度抑制（仅当货箱接近泊位时门控） ----
    dock_speed_excess = crate_speed - 0.05
    if dock_speed_excess < 0.0:
        dock_speed_excess = 0.0
    crate_speed_near_dock = -1.0 * near_gate * dock_speed_excess

    # ---- 7) 越界惩罚（hinge，仅在接近边界时生效） ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 小车位置归一化到 [-1,1] 附近，超过 0.85 认为接近边界
    cart_edge = 0.0
    ax = cart_x if cart_x >= 0.0 else -cart_x
    ay = cart_y if cart_y >= 0.0 else -cart_y
    if ax > 0.85:
        cart_edge = cart_edge + (ax - 0.85)
    if ay > 0.85:
        cart_edge = cart_edge + (ay - 0.85)
    # 货箱距泊位偏移的绝对值过大也视为远离安全区（间接反映越界风险）
    dist_edge = dist - 1.5
    if dist_edge < 0.0:
        dist_edge = 0.0
    out_of_bounds_penalty = -2.0 * cart_edge - 1.0 * dist_edge

    # ---- 8) 障碍接近惩罚（hinge，仅在非常接近时生效） ----
    obstacle = 0.0
    if front_prox > 0.7:
        obstacle = obstacle + (front_prox - 0.7)
    if left_prox > 0.7:
        obstacle = obstacle + (left_prox - 0.7)
    if right_prox > 0.7:
        obstacle = obstacle + (right_prox - 0.7)
    obstacle_penalty = -0.3 * obstacle

    # ---- 9) 动作平滑（轻量，仅抑制极端抖动） ----
    drive = action[0]
    steer = action[1]
    action_smoothness = -0.02 * (drive * drive + steer * steer)

    # ---- 汇总 ----
    components = {}
    components["crate_to_dock_progress"] = crate_to_dock_progress
    components["crate_dock_proximity"] = crate_dock_proximity
    components["crate_docking_quality"] = crate_docking_quality
    components["crate_inside_bonus"] = crate_inside_bonus
    components["soft_contact_penalty"] = soft_contact_penalty
    components["crate_speed_near_dock"] = crate_speed_near_dock
    components["out_of_bounds_penalty"] = out_of_bounds_penalty
    components["obstacle_penalty"] = obstacle_penalty
    components["action_smoothness"] = action_smoothness

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]

    return (float(total_reward), components)