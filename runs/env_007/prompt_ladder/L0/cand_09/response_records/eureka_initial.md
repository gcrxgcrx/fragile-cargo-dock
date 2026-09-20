# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 常量 ----
    DOCK_X_TOL = 0.024      # 完全进入泊位的 x 偏移阈值
    DOCK_Y_TOL = 0.030      # 完全进入泊位的 y 偏移阈值
    DOCK_X_SOFT = 0.12      # 接近泊位的软阈值
    DOCK_Y_SOFT = 0.15
    ANGLE_TOL = 0.5236      # 30 度
    SPEED_TOL = 0.05        # 静止阈值 (m/s 归一化前)
    CART_BOUND = 0.92       # 小车越界软阈值(归一化坐标)
    CRATE_BOUND = 0.92      # 货箱越界软阈值(由恢复坐标估计)

    # ---- 观测解包 ----
    cart_x = obs[0]
    cart_y = obs[1]
    cart_cos = obs[2]
    cart_sin = obs[3]
    cart_fwd_speed = obs[4]
    cart_yaw = obs[5]
    crate_rel_x_body = obs[6]
    crate_rel_y_body = obs[7]
    crate_vx = obs[8]
    crate_vy = obs[9]
    crate_cos = obs[10]
    crate_sin = obs[11]
    crate_dock_x = obs[12]
    crate_dock_y = obs[13]
    cart_crate_contact = obs[14]
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    time_fraction = obs[18]

    n_cart_x = next_obs[0]
    n_cart_y = next_obs[1]
    n_crate_dock_x = next_obs[12]
    n_crate_dock_y = next_obs[13]

    # ---- 1. crate_to_dock_progress: 用 delta(distance) 驱动货箱靠近泊位 ----
    # 归一化到 [-2,2] 的坐标，真实米制距离可用半宽/半高恢复，但相对比较用归一化即可
    cur_dist = (crate_dock_x ** 2 + crate_dock_y ** 2) ** 0.5
    next_dist = (n_crate_dock_x ** 2 + n_crate_dock_y ** 2) ** 0.5
    progress = cur_dist - next_dist  # 正=靠近
    # 平滑压缩，避免单步过大
    progress_component = 2.0 * (progress / (1.0 + abs(progress)))

    # ---- 2. crate_docking_quality: 位置/朝向/速度联合代理 ----
    # 位置因子：越接近泊位中心越接近 1
    pos_err = (crate_dock_x ** 2 + crate_dock_y ** 2) ** 0.5
    pos_factor = max(0.0, 1.0 - pos_err / 0.20)

    # 朝向因子：货箱朝向误差 (atan2 -> cos 误差)
    # 货箱朝向对齐泊位，泊位朝向假定为 0（cos=1,sin=0）
    crate_angle_err = abs(crate_sin)  # sin 分量近似角度误差大小
    angle_factor = max(0.0, 1.0 - crate_angle_err / (ANGLE_TOL + 0.3))

    # 速度因子：越静止越接近 1
    crate_speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 8.0 * crate_speed)

    # 几何平均联合代理，避免乘积塌缩
    quality = (pos_factor * angle_factor * speed_factor) ** (1.0 / 3.0)
    quality_component = 3.0 * quality

    # ---- 3. crate_speed_penalty_near_dock: 仅在接近泊位时抑制货箱速度 ----
    near_dock = max(0.0, 1.0 - pos_err / 0.15)
    speed_penalty = -1.5 * near_dock * (crate_speed ** 2)

    # ---- 4. soft_contact_penalty: 接触时速度突变间接推断硬碰撞 ----
    # 用当前货箱速度与小车速度差作为间接指标；接触时若货箱速度大则可能硬碰撞
    if cart_crate_contact > 0.5:
        rel_speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5
        # 只在货箱速度超过软阈值时惩罚
        contact_penalty = -0.8 * max(0.0, rel_speed - 0.15)
    else:
        contact_penalty = 0.0

    # ---- 5. out_of_bounds_penalty: 小车与货箱接近边界时 hinge 惩罚 ----
    cart_margin = max(abs(cart_x), abs(cart_y))
    cart_oob = -3.0 * max(0.0, cart_margin - CART_BOUND)

    # 恢复货箱世界坐标（近似）
    # 车体系 -> 世界系: 旋转 cart_cos, cart_sin
    crate_wx = (cart_cos * crate_rel_x_body - cart_sin * crate_rel_y_body) * 3.0 + cart_x * 5.0
    crate_wy = (cart_sin * crate_rel_x_body + cart_cos * crate_rel_y_body) * 3.0 + cart_y * 4.0
    # 归一化估计（半宽 5，半高 4）
    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0
    crate_margin = max(abs(crate_nx), abs(crate_ny))
    crate_oob = -3.0 * max(0.0, crate_margin - CRATE_BOUND)

    oob_component = cart_oob + crate_oob

    # ---- 6. action_smoothness: 轻量动作惩罚 ----
    smooth_penalty = -0.05 * (action[0] ** 2 + action[1] ** 2)

    # ---- 7. obstacle_penalty: 前方障碍接近度 ----
    obstacle_penalty = -0.3 * max(0.0, sensor_front - 0.7)

    components = {
        "crate_to_dock_progress": float(progress_component),
        "crate_docking_quality": float(quality_component),
        "crate_speed_penalty_near_dock": float(speed_penalty),
        "soft_contact_penalty": float(contact_penalty),
        "out_of_bounds_penalty": float(oob_component),
        "action_smoothness": float(smooth_penalty),
        "obstacle_penalty": float(obstacle_penalty),
    }

    total_reward = sum(components.values())
    return float(total_reward), components
```
