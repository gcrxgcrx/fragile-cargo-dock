# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 常量 ----
    # 泊位进入阈值（卡片给出）
    DOCK_X_TOL = 0.024
    DOCK_Y_TOL = 0.030
    # 朝向对齐阈值：30 度
    ANGLE_TOL = 0.5235987755982988
    # 静止阈值：0.05 m/s，obs 中速度已 /3.0
    SPEED_TOL = 0.05 / 3.0

    # ============ 1. 货箱到泊位进度（主信号，用 improvement delta） ============
    # 用归一化偏移构造距离度量（obs[12] x 半宽, obs[13] y 半高）
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 距离减少为正，放大以形成有效梯度
    progress = (cur_dist - nxt_dist) * 8.0

    # ============ 2. 泊位接近度（有界，鼓励收敛到泊位中心） ============
    # 平滑压缩，避免悬停陷阱：距离越小奖励越高，但有界
    proximity = 1.0 / (1.0 + 6.0 * nxt_dist)

    # ============ 3. 朝向对齐 shaping ============
    # 货箱朝向误差 = atan2(sin, cos) 的绝对值，用 cos 值近似对齐度
    # cos(heading) 越接近 1 表示对齐越好（朝向 0 度），但需考虑 ±pi 对称
    crate_cos = next_obs[10]
    # 对齐因子：cos 值在 [-1,1]，映射到 [0,1]
    align_factor = (crate_cos + 1.0) * 0.5

    # ============ 4. 货箱速度（接近泊位时抑制） ============
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    # 归一化速度（假定最大约 3 m/s 量级）
    crate_speed_norm = crate_speed / 3.0

    # 接近泊位时启用速度抑制（用 proximity 作为门）
    near_dock_gate = 1.0 / (1.0 + 8.0 * nxt_dist)
    speed_penalty = -0.5 * near_dock_gate * (crate_speed_norm ** 2)

    # ============ 5. 联合完成代理（进入 + 对齐 + 静止） ============
    # 连续 bounded factor
    # 位置因子：距离越近越接近 1
    pos_factor = 1.0 / (1.0 + 30.0 * nxt_dist)
    # 对齐因子
    align_proxy = align_factor
    # 静止因子：速度越小越接近 1
    still_factor = 1.0 / (1.0 + 40.0 * crate_speed_norm)

    # 几何平均，避免乘积塌缩
    joint_proxy = (pos_factor * align_proxy * still_factor) ** (1.0 / 3.0)

    # ============ 6. 轻柔接触 / 硬碰撞间接惩罚 ============
    # 接触时若货箱速度突变大，间接推断可能有硬碰撞
    contact = next_obs[14]
    # 货箱速度变化
    prev_crate_vx = obs[8] * 3.0
    prev_crate_vy = obs[9] * 3.0
    dvx = crate_vx - prev_crate_vx
    dvy = crate_vy - prev_crate_vy
    dv_mag = (dvx * dvx + dvy * dvy) ** 0.5
    # 仅在接触时，速度突变超过阈值才罚（hinge）
    soft_contact_penalty = 0.0
    if contact > 0.5:
        excess = dv_mag - 1.0
        if excess > 0.0:
            soft_contact_penalty = -0.3 * excess

    # ============ 7. 越界惩罚（hinge） ============
    cart_x = obs[0]
    cart_y = obs[1]
    # 小车越界：|x|>0.9 或 |y|>0.9 附近开始惩罚
    out_penalty = 0.0
    cart_excess_x = abs(cart_x) - 0.9
    if cart_excess_x > 0.0:
        out_penalty -= 1.0 * cart_excess_x
    cart_excess_y = abs(cart_y) - 0.9
    if cart_excess_y > 0.0:
        out_penalty -= 1.0 * cart_excess_y
    # 货箱越界：用归一化偏移估计（泊位偏移过大意味着货箱远离，但不等价越界）
    crate_excess = nxt_dist - 1.5
    if crate_excess > 0.0:
        out_penalty -= 1.0 * crate_excess

    # ============ 8. 障碍接近惩罚（hinge，防撞墙） ============
    obs_penalty = 0.0
    front = next_obs[15]
    left = next_obs[16]
    right = next_obs[17]
    if front > 0.85:
        obs_penalty -= 0.2 * (front - 0.85)
    if left > 0.85:
        obs_penalty -= 0.2 * (left - 0.85)
    if right > 0.85:
        obs_penalty -= 0.2 * (right - 0.85)

    # ============ 9. 动作平滑（轻量，防止抖动） ============
    drive = action[0]
    steer = action[1]
    action_penalty = -0.02 * (drive * drive + steer * steer)

    # ============ 组装 ============
    comp_dock_progress = progress * 1.0
    comp_dock_proximity = proximity * 1.5
    comp_dock_quality = joint_proxy * 2.0
    comp_speed_penalty = speed_penalty
    comp_soft_contact = soft_contact_penalty
    comp_out_of_bounds = out_penalty
    comp_obstacle = obs_penalty
    comp_action_smooth = action_penalty

    total = (
        comp_dock_progress
        + comp_dock_proximity
        + comp_dock_quality
        + comp_speed_penalty
        + comp_soft_contact
        + comp_out_of_bounds
        + comp_obstacle
        + comp_action_smooth
    )

    components = {
        "crate_to_dock_progress": float(comp_dock_progress),
        "crate_dock_proximity": float(comp_dock_proximity),
        "crate_docking_quality": float(comp_dock_quality),
        "crate_speed_penalty_near_dock": float(comp_speed_penalty),
        "soft_contact_penalty": float(comp_soft_contact),
        "out_of_bounds_penalty": float(comp_out_of_bounds),
        "obstacle_proximity_penalty": float(comp_obstacle),
        "action_smoothness_penalty": float(comp_action_smooth),
    }

    return float(total), components
```
