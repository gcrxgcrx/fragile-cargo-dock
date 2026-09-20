```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取（仅使用声明的 obs 维度） ----
    # 货箱到 dock 偏移（归一化），转成米近似
    cur_dx = obs[12] * 5.0
    cur_dy = obs[13] * 4.0
    nxt_dx = next_obs[12] * 5.0
    nxt_dy = next_obs[13] * 4.0

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 货箱速率 (m/s)
    cur_crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5
    nxt_crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向误差（相对 dock 对齐，dock 朝向设为 0 轴）
    crate_heading = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_heading < 1e-6:
        angle_err = 0.0
    else:
        # 用 cos 值近似对齐程度：|cos(theta)| 越接近 1 越对齐
        cos_h = next_obs[10] / crate_heading
        angle_err = 1.0 - abs(cos_h)  # 0 表示完全对齐

    # 接触标志
    contact = next_obs[14]

    # 小车速度/角速度
    cart_speed = next_obs[4] * 3.0
    cart_yaw = next_obs[5] * 8.0

    # 时间比例
    time_frac = next_obs[18]
    if time_frac < 0.0:
        time_frac = 0.0
    if time_frac > 1.0:
        time_frac = 1.0

    # 障碍接近度（前方）
    front_prox = next_obs[15]

    # ---- 组件 1: crate_to_dock_progress ----
    # 距离减少量，正向奖励靠近；用 improvement_delta
    delta_dist = cur_dist - nxt_dist
    # 限制单步幅度，防止刷分
    if delta_dist > 1.0:
        delta_dist = 1.0
    if delta_dist < -1.0:
        delta_dist = -1.0
    progress_reward = 2.0 * delta_dist

    # 接近程度（bounded）：鼓励靠近 dock
    near_factor = 1.0 / (1.0 + 0.5 * nxt_dist)

    # ---- 组件 2: crate_dock_alignment ----
    # 仅在货箱接近 dock 时启用（距离 < 2m）
    if nxt_dist < 2.0:
        align_gate = 1.0 - nxt_dist / 2.0
        # cos 对齐误差，越对齐越大
        align_signal = (1.0 - angle_err) * align_gate
    else:
        align_signal = 0.0
    alignment_reward = 0.8 * align_signal

    # ---- 组件 3: crate_settling ----
    # 货箱接近 dock 时，速度越低越好
    if nxt_dist < 1.5:
        settle_gate = 1.0 - nxt_dist / 1.5
        # 速度低于 0.5 m/s 时给奖励，高于则衰减
        speed_factor = 1.0 / (1.0 + 4.0 * nxt_crate_speed)
        settling_reward = 0.8 * settle_gate * speed_factor
    else:
        settling_reward = 0.0

    # ---- 组件 4: contact_gated_push ----
    # 接触且货箱未进入 dock 时，鼓励有效推动（推进方向与到 dock 方向一致）
    if contact > 0.5 and nxt_dist > 0.3:
        # 推动有效性：货箱速度朝向 dock 方向的分量
        if cur_dist > 1e-6:
            dir_x = -cur_dx / cur_dist
            dir_y = -cur_dy / cur_dist
        else:
            dir_x = 0.0
            dir_y = 0.0
        push_vx = next_obs[8] * 3.0
        push_vy = next_obs[9] * 3.0
        push_align = push_vx * dir_x + push_vy * dir_y
        if push_align < 0.0:
            push_align = 0.0
        # 限制幅度
        if push_align > 1.5:
            push_align = 1.5
        push_reward = 0.5 * push_align
    else:
        push_reward = 0.0

    # ---- 组件 5: impact_softness ----
    # 接触时速度突变过大 -> 惩罚（近似硬冲击）
    impact_penalty = 0.0
    if contact > 0.5:
        crate_speed_jump = nxt_crate_speed - cur_crate_speed
        if crate_speed_jump > 0.0:
            # 速度增加说明被撞加速，冲击风险
            impact_penalty = -0.3 * crate_speed_jump
        # 高速接触本身也抑制
        if nxt_crate_speed > 1.0:
            impact_penalty -= 0.2 * (nxt_crate_speed - 1.0)

    # ---- 组件 6: boundary_avoidance ----
    # 小车位置越界风险（|cart_x|>1 或 |cart_y|>1 为出界）
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    boundary_penalty = 0.0
    # 接近边界时 hinge 惩罚
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    # 前方障碍接近度
    if front_prox > 0.8:
        boundary_penalty -= 0.3 * (front_prox - 0.8)

    # 货箱世界坐标反推，检查货箱越界风险
    # 货箱相对小车体坐标转世界坐标
    cart_cos = next_obs[2]
    cart_sin = next_obs[3]
    rel_bx = next_obs[6] * 3.0
    rel_by = next_obs[7] * 3.0
    crate_wx = cart_cos * rel_bx - cart_sin * rel_by + cart_x * 5.0
    crate_wy = cart_sin * rel_bx + cart_cos * rel_by + cart_y * 4.0
    # 货箱越界风险（仓库半宽约 5m，半高约 4m，留余量）
    if abs(crate_wx) > 4.3:
        boundary_penalty -= 0.5 * (abs(crate_wx) - 4.3)
    if abs(crate_wy) > 3.4:
        boundary_penalty -= 0.5 * (abs(crate_wy) - 3.4)

    # ---- 组件 7: joint_condition_proxy ----
    # 软完成近似：位置到位 + 朝向对齐 + 低速 的联合因子
    # 位置因子
    pos_factor = 1.0 / (1.0 + 3.0 * nxt_dist)
    # 朝向因子
    align_factor = 1.0 - angle_err
    # 速度因子
    vel_factor = 1.0 / (1.0 + 20.0 * nxt_crate_speed)
    # 几何平均，避免塌缩
    joint_proxy = (pos_factor * align_factor * vel_factor) ** (1.0 / 3.0)
    joint_reward = 1.5 * joint_proxy

    # ---- 组件 8: action_smoothness ----
    # 轻量动作惩罚，避免剧烈抖动（不过度压制）
    drive = action[0]
    steer = action[1]
    action_penalty = -0.05 * (drive * drive + steer * steer)

    # ---- 总奖励 ----
    components = {}
    components["crate_to_dock_progress"] = float(progress_reward)
    components["crate_near_dock"] = float(0.5 * near_factor)
    components["crate_dock_alignment"] = float(alignment_reward)
    components["crate_settling"] = float(settling_reward)
    components["contact_gated_push"] = float(push_reward)
    components["impact_softness"] = float(impact_penalty)
    components["boundary_avoidance"] = float(boundary_penalty)
    components["joint_condition_proxy"] = float(joint_reward)
    components["action_smoothness"] = float(action_penalty)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components
```