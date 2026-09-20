```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主信号：货箱到泊位的距离缩减（improvement_delta） ----
    dx_old = obs[12]
    dy_old = obs[13]
    dx_new = next_obs[12]
    dy_new = next_obs[13]

    dist_old = (dx_old * dx_old + dy_old * dy_old) ** 0.5
    dist_new = (dx_new * dx_new + dy_new * dy_new) ** 0.5

    progress = dist_old - dist_new  # 正=靠近泊位

    # ---- 货箱接近度（bounded，作为进入泊位前的连续引导，不压过 delta）----
    proximity = 1.0 / (1.0 + 6.0 * dist_new)

    # ---- 货箱速度（世界系，恢复米制量级）----
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    speed_norm = crate_speed / 3.0
    if speed_norm > 1.0:
        speed_norm = 1.0

    # ---- 货箱朝向误差 ----
    crate_heading_err = (next_obs[10] * next_obs[10] + next_obs[11] * next_obs[11]) ** 0.5
    if crate_heading_err < 1e-6:
        crate_heading_err = 1e-6
    cos_err = next_obs[10] / crate_heading_err
    if cos_err > 1.0:
        cos_err = 1.0
    if cos_err < -1.0:
        cos_err = -1.0
    heading_factor = (1.0 + cos_err) * 0.5  # 0=反向, 1=对齐

    # ---- 门控：仅在货箱接近泊位时激活停靠质量项 ----
    # 距离阈值：进入泊位附近（dist < 0.35 归一化）才开始
    near_gate = 1.0 / (1.0 + 12.0 * dist_new)  # 0..1，越近越大

    # 静止因子：速度越小越接近 1
    still_factor = 1.0 / (1.0 + 8.0 * speed_norm)

    # 联合停靠质量：位置(proximity) * 朝向 * 静止，用几何平均避免塌缩
    dock_quality = (proximity * heading_factor * still_factor) ** (1.0 / 3.0)

    # ---- 越界风险 hinge 惩罚（小车与货箱坐标）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_margin = 0.92
    oob_cart = 0.0
    if cart_x > cart_margin:
        oob_cart += (cart_x - cart_margin)
    if cart_x < -cart_margin:
        oob_cart += (-cart_margin - cart_x)
    if cart_y > cart_margin:
        oob_cart += (cart_y - cart_margin)
    if cart_y < -cart_margin:
        oob_cart += (-cart_margin - cart_y)

    # 货箱世界坐标恢复
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_wx = (cart_x * 5.0) + (rel_x * cos_h - rel_y * sin_h)
    crate_wy = (cart_y * 4.0) + (rel_x * sin_h + rel_y * cos_h)
    crate_margin_x = 4.6
    crate_margin_y = 3.6
    oob_crate = 0.0
    if crate_wx > crate_margin_x:
        oob_crate += (crate_wx - crate_margin_x)
    if crate_wx < -crate_margin_x:
        oob_crate += (-crate_margin_x - crate_wx)
    if crate_wy > crate_margin_y:
        oob_crate += (crate_wy - crate_margin_y)
    if crate_wy < -crate_margin_y:
        oob_crate += (-crate_margin_y - crate_wy)

    # ---- 接触时的激进操作惩罚（仅在接触且相对速度高时）----
    contact = next_obs[14]
    cart_speed = abs(next_obs[4])  # 归一化前向速度
    contact_aggression = 0.0
    if contact > 0.5:
        # 接触时货箱速度高 → 可能硬碰撞
        if speed_norm > 0.25:
            contact_aggression = (speed_norm - 0.25) * (speed_norm - 0.25)

    # ---- 组件汇总 ----
    progress_reward = 25.0 * progress
    proximity_reward = 0.6 * proximity
    dock_quality_reward = 1.5 * near_gate * dock_quality
    oob_penalty = -3.0 * (oob_cart + oob_crate)
    contact_penalty = -0.8 * contact_aggression
    action_cost = -0.02 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": progress_reward,
        "crate_proximity": proximity_reward,
        "crate_docking_quality": dock_quality_reward,
        "out_of_bounds_penalty": oob_penalty,
        "soft_contact_penalty": contact_penalty,
        "action_cost": action_cost,
    }

    total_reward = (
        progress_reward
        + proximity_reward
        + dock_quality_reward
        + oob_penalty
        + contact_penalty
        + action_cost
    )

    return (float(total_reward), components)
```