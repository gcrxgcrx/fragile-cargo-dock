```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主信号 1: 货箱到泊位的进度（用 delta 距离，避免悬停陷阱）----
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]
    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5
    progress = cur_dist - nxt_dist
    crate_to_dock_progress = 12.0 * progress

    # ---- 主信号 2: 泊位接近度（有界，作为稠密引导）----
    proximity = 1.0 / (1.0 + 6.0 * nxt_dist)
    crate_dock_proximity = 1.0 * proximity

    # ---- 主信号 3: 停靠质量联合代理（进入 + 朝向对齐 + 静止）----
    # 位置因子：越接近泊位中心越接近 1
    pos_factor = 1.0 / (1.0 + 25.0 * nxt_dist)
    # 朝向因子：货箱朝向误差（cos 越大越对齐）
    crate_cos = next_obs[10]
    align_factor = (crate_cos + 1.0) * 0.5
    if align_factor < 0.0:
        align_factor = 0.0
    if align_factor > 1.0:
        align_factor = 1.0
    # 速度因子：货箱速度越小越接近 1
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 8.0 * crate_speed)
    docking_quality = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 2.5 * docking_quality

    # ---- 主信号 4: 货箱进入泊位的稀疏奖励（几何阈值重建）----
    inside = 0.0
    if abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030:
        inside = 1.0
    crate_inside_dock = 3.0 * inside

    # ---- 约束 1: 接近泊位时的货箱速度抑制（门控式，只在近泊位时激活）----
    near_gate = 1.0 / (1.0 + 20.0 * nxt_dist)
    crate_speed_penalty = -1.5 * near_gate * crate_speed

    # ---- 约束 2: 软接触惩罚（接触且货箱速度突变时，轻量抑制）----
    contact = next_obs[14]
    # 用货箱速度与小车速度差近似相对速度
    cart_speed = abs(next_obs[4]) * 3.0
    rel_speed = crate_speed + cart_speed
    hard_like = 0.0
    if contact > 0.5 and rel_speed > 1.0:
        hard_like = rel_speed - 1.0
    soft_contact_penalty = -0.8 * hard_like

    # ---- 约束 3: 越界惩罚（hinge，只在接近边界时生效）----
    cart_x = obs[0]
    cart_y = obs[1]
    # 恢复货箱世界坐标（近似：用车体系相对位置 + 小车位置）
    cart_cos_h = obs[2]
    cart_sin_h = obs[3]
    crate_rel_x = obs[6] * 3.0
    crate_rel_y = obs[7] * 3.0
    crate_wx = cart_x * 5.0 + crate_rel_x * cart_cos_h - crate_rel_y * cart_sin_h
    crate_wy = cart_y * 4.0 + crate_rel_x * cart_sin_h + crate_rel_y * cart_cos_h
    out_pen = 0.0
    cart_margin = max(abs(cart_x) - 0.9, abs(cart_y) - 0.9, 0.0)
    out_pen += cart_margin * cart_margin
    crate_margin = max(abs(crate_wx) / 5.0 - 0.9, abs(crate_wy) / 4.0 - 0.9, 0.0)
    out_pen += crate_margin * crate_margin
    out_of_bounds_penalty = -6.0 * out_pen

    # ---- 约束 4: 障碍接近惩罚（hinge，只在很近时激活）----
    obs_front = next_obs[15]
    obs_left = next_obs[16]
    obs_right = next_obs[17]
    obst_pen = 0.0
    if obs_front > 0.85:
        obst_pen += (obs_front - 0.85) ** 2
    if obs_left > 0.85:
        obst_pen += (obs_left - 0.85) ** 2
    if obs_right > 0.85:
        obst_pen += (obs_right - 0.85) ** 2
    obstacle_penalty = -2.0 * obst_pen

    # ---- 约束 5: 动作平滑（轻量，仅抑制剧烈抖动）----
    action_smoothness = -0.05 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_dock_proximity": float(crate_dock_proximity),
        "crate_docking_quality": float(crate_docking_quality),
        "crate_inside_dock": float(crate_inside_dock),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total = 0.0
    for key in components:
        total += components[key]

    return float(total), components
```