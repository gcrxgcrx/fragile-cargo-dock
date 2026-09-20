```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 几何与阈值常数（来自环境事实） ----
    # 泊位容差：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030
    # 朝向误差 < 30 度；速度 < 0.05 m/s；连续保持 10 步
    dock_x_tol = 0.024
    dock_y_tol = 0.030
    speed_tol = 0.05
    ang_tol_rad = 0.5235987756  # 30 度

    # ---- 当前量 ----
    dx_now = obs[12]
    dy_now = obs[13]
    dx_next = next_obs[12]
    dy_next = next_obs[13]

    # 米制距离（半宽 5.0 m, 半高 4.0 m，来自专家卡片 derived_possible）
    dist_now = ((dx_now * 5.0) ** 2 + (dy_now * 4.0) ** 2) ** 0.5
    dist_next = ((dx_next * 5.0) ** 2 + (dy_next * 4.0) ** 2) ** 0.5

    # 货箱速度（世界系，m/s）
    vx = next_obs[8] * 3.0
    vy = next_obs[9] * 3.0
    crate_speed = (vx * vx + vy * vy) ** 0.5

    # 货箱朝向误差（弧度），归一化到 [0, pi]
    crate_ang = (next_obs[11] ** 2 + next_obs[10] ** 2) ** 0.5
    # 泊位朝向对齐：货箱朝向与目标朝向（假设泊位朝向为 0 度，cos=1）
    # 使用 |sin| 与 |1-cos| 组合近似角度误差，避免 atan2
    cos_h = next_obs[10]
    sin_h = next_obs[11]
    # 角度误差的连续近似
    ang_err = (sin_h * sin_h + (1.0 - cos_h) * (1.0 - cos_h)) ** 0.5

    # ---- 组件 1：货箱向泊位的增量进展（主信号） ----
    # 只奖励"这一帧更接近了"，避免悬停收割
    progress = dist_now - dist_next
    # 放大到合理量级：0.1 m 的改进 → 0.1 * scale
    crate_to_dock_progress = 30.0 * progress

    # ---- 组件 2：接近泊位时的速度抑制（门控，仅在接近时启用） ----
    # 门控因子：距离泊位越近，门越大
    near_gate = max(0.0, 1.0 - dist_next / 1.0)  # 1 m 内开始生效
    speed_pen = -2.0 * near_gate * (crate_speed ** 2)
    crate_speed_penalty_near_dock = speed_pen

    # ---- 组件 3：接近泊位时的朝向对齐 shaping ----
    # 仅在接近泊位时激活，避免全局持续惩罚
    align_pen = -1.5 * near_gate * (ang_err ** 2)
    crate_orientation_align = align_pen

    # ---- 组件 4：软接触惩罚（间接：接触 + 货箱高速） ----
    contact = next_obs[14]
    # 接触时货箱速度过高 → 可能硬碰撞
    contact_excess = max(0.0, crate_speed - 0.3)
    soft_contact_penalty = -3.0 * contact * contact_excess

    # ---- 组件 5：越界惩罚（小车与货箱接近仓库边界） ----
    # 小车位置归一化到 [-1,1]，边界为 1.0；货箱位置由 dx, dy 恢复
    cart_x = obs[0]
    cart_y = obs[1]
    crate_world_x = cart_x * 5.0 + (obs[6] * 3.0)
    crate_world_y = cart_y * 4.0 + (obs[7] * 3.0)
    # 仓库半宽 5.0, 半高 4.0
    cart_edge = max(abs(cart_x), abs(cart_y))
    crate_edge_x = abs(crate_world_x) / 5.0
    crate_edge_y = abs(crate_world_y) / 4.0
    crate_edge = max(crate_edge_x, crate_edge_y)
    # hinge：仅在超过 0.85 时惩罚
    cart_oob = max(0.0, cart_edge - 0.85)
    crate_oob = max(0.0, crate_edge - 0.85)
    out_of_bounds_penalty = -20.0 * (cart_oob ** 2 + crate_oob ** 2)

    # ---- 组件 6：动作平滑（轻量） ----
    action_smoothness = -0.05 * (action[0] ** 2 + action[1] ** 2)

    # ---- 组件 7：完成事件（一次性大额奖励） ----
    # 条件：货箱完全在泊位内 + 朝向对齐 + 速度低
    inside_dock = (abs(dx_next) <= dock_x_tol) and (abs(dy_next) <= dock_y_tol)
    aligned = ang_err < (ang_tol_rad * 0.5)  # 更严格的对齐
    slow = crate_speed < speed_tol
    if inside_dock and aligned and slow:
        completion_event = 500.0
    else:
        completion_event = 0.0

    # ---- 汇总 ----
    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "crate_orientation_align": crate_orientation_align,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
        "completion_event": completion_event,
    }

    total_reward = (
        crate_to_dock_progress
        + crate_speed_penalty_near_dock
        + crate_orientation_align
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
        + completion_event
    )

    return float(total_reward), components
```