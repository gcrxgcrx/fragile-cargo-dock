```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 从观测恢复关键量 ----
    # 货箱到泊位的有符号偏移（归一化）
    dock_dx = next_obs[12]
    dock_dy = next_obs[13]
    dock_dist = (dock_dx * dock_dx + dock_dy * dock_dy) ** 0.5

    old_dock_dx = obs[12]
    old_dock_dy = obs[13]
    old_dock_dist = (old_dock_dx * old_dock_dx + old_dock_dy * old_dock_dy) ** 0.5

    # 货箱速度（归一化），真实速度 = obs*3.0 m/s
    crate_vx = next_obs[8]
    crate_vy = next_obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向误差（弧度）
    cos_h = next_obs[10]
    sin_h = next_obs[11]
    # 归一化到 [-1,1] 防止数值越界
    if cos_h > 1.0:
        cos_h = 1.0
    if cos_h < -1.0:
        cos_h = -1.0
    # 朝向误差用 |sin| 近似（对齐时 sin->0），等价于角度误差的单调度量
    heading_err = abs(sin_h)

    # 小车与货箱接触
    contact = next_obs[14]

    # ---- 组件 1：货箱向泊位推进（delta 形式，避免悬停陷阱）----
    progress = old_dock_dist - dock_dist
    crate_to_dock_progress = 12.0 * progress

    # ---- 组件 2：货箱接近泊位（bounded，防悬停用的稠密吸引）----
    # 距离越小奖励越大，用倒数衰减保持有界
    approach = 1.0 / (1.0 + 6.0 * dock_dist)
    crate_approach = 1.5 * approach

    # ---- 组件 3：进入泊位后的复合停靠质量（位置 + 朝向 + 静止）----
    # 位置因子：越靠近泊位中心越接近 1
    pos_factor = 1.0 / (1.0 + 40.0 * dock_dist)
    # 朝向因子：sin 误差越小越接近 1
    align_factor = 1.0 / (1.0 + 8.0 * heading_err)
    # 静止因子：货箱速度越小越接近 1
    still_factor = 1.0 / (1.0 + 30.0 * crate_speed)
    # 几何平均，避免乘积塌缩
    dock_quality = (pos_factor * align_factor * still_factor) ** (1.0 / 3.0)
    # 只在接近泊位时显著生效（用 pos_factor 作为软门）
    crate_docking_quality = 6.0 * dock_quality * pos_factor

    # ---- 组件 4：接近泊位时的速度抑制（hinge，仅在近泊位时启用）----
    # 近泊位权重：距离 < 0.15 时开始生效
    near_weight = 0.0
    if dock_dist < 0.15:
        near_weight = (0.15 - dock_dist) / 0.15
    speed_excess = crate_speed - 0.02
    if speed_excess < 0.0:
        speed_excess = 0.0
    crate_speed_penalty_near_dock = -8.0 * near_weight * (speed_excess ** 2)

    # ---- 组件 5：轻柔接触（接触时惩罚过大的相对速度）----
    # 用货箱速度近似冲击强度，接触时抑制高速推撞
    soft_contact_penalty = 0.0
    if contact > 0.5:
        fast = crate_speed - 0.15
        if fast > 0.0:
            soft_contact_penalty = -3.0 * (fast ** 2)

    # ---- 组件 6：越界防护（hinge，仅在小车/货箱接近边界时生效）----
    # 小车位置 obs[0], obs[1] 归一化到约 [-1,1]
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    out_of_bounds_penalty = 0.0
    # 小车越界 hinge
    cart_ax = abs(cart_x)
    if cart_ax > 0.85:
        out_of_bounds_penalty -= 4.0 * ((cart_ax - 0.85) ** 2)
    cart_ay = abs(cart_y)
    if cart_ay > 0.85:
        out_of_bounds_penalty -= 4.0 * ((cart_ay - 0.85) ** 2)

    # ---- 组件 7：动作平滑（轻量，抑制剧烈抖动）----
    drive = action[0]
    steer = action[1]
    action_smoothness = -0.05 * (drive * drive + steer * steer)

    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_approach": crate_approach,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```