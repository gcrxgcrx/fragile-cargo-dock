# Response Record

# 分析

问题非常明确：`joint_condition_proxy` 每步无条件激活（100%），episode sum 约 425，几乎等于 total_reward（417），完全压过其余所有组件。它的 `pos_factor = 1/(1+5*dist)` 在货箱远离 dock 时仍有 ~0.2 的基线，几何平均后每步稳定给 ~1.0 的正奖励——**agent 只要活着不动就能刷分**，导致 400 步全部 truncation、0 次成功、任务分数 -1.34。而真正驱动任务的组件（progress/push/alignment/settling）激活率 <0.3%，几乎从未触发，说明策略根本没学会接触推动。

改进方向：(1) 把 joint_proxy 改成**只在货箱接近 dock 时才有意义**的联合完成近似，并大幅降权，使其不再主导；(2) 用**距离势能差**（potential delta）替代稀疏的接触门控 delta，提供稠密进度梯度；(3) 强化 alignment/settling 的门控范围与权重，让"进 dock 后停稳对齐"有明确信号；(4) 保留动作平滑与边界约束但降权，避免压制探索。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取（仅使用声明的 obs 维度） ----
    cur_dx = obs[12] * 5.0
    cur_dy = obs[13] * 4.0
    nxt_dx = next_obs[12] * 5.0
    nxt_dy = next_obs[13] * 4.0

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 货箱速率 (m/s)
    cur_crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5
    nxt_crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐程度：|cos(theta)| 越接近 1 越对齐
    crate_heading = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_heading < 1e-6:
        align_cos = 0.0
    else:
        align_cos = abs(next_obs[10] / crate_heading)
    angle_err = 1.0 - align_cos

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

    # 障碍接近度
    front_prox = next_obs[15]

    # ---- 组件 1: crate_to_dock_progress（势能差，稠密） ----
    # 用距离减少量作为稠密进度信号；不再用接触硬门控，避免稀疏
    delta_dist = cur_dist - nxt_dist
    if delta_dist > 0.5:
        delta_dist = 0.5
    if delta_dist < -0.5:
        delta_dist = -0.5
    progress_reward = 8.0 * delta_dist

    # ---- 组件 2: contact_gated_push ----
    # 接触时，货箱速度朝向 dock 的分量越大越好
    if contact > 0.5 and nxt_dist > 0.2:
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
        if push_align > 1.5:
            push_align = 1.5
        push_reward = 1.5 * push_align
    else:
        push_reward = 0.0

    # ---- 组件 3: crate_dock_alignment ----
    # 货箱接近 dock 时，朝向对齐给奖励；门控放宽到 4.0m 并提高权重
    if nxt_dist < 4.0:
        align_gate = 1.0 - nxt_dist / 4.0
        alignment_reward = 3.0 * align_gate * align_cos
    else:
        alignment_reward = 0.0

    # ---- 组件 4: crate_settling ----
    # 货箱接近 dock 时，速度越低越好；门控放宽到 3.0m 并提高权重
    if nxt_dist < 3.0:
        settle_gate = 1.0 - nxt_dist / 3.0
        speed_factor = 1.0 / (1.0 + 4.0 * nxt_crate_speed)
        settling_reward = 3.0 * settle_gate * speed_factor
    else:
        settling_reward = 0.0

    # ---- 组件 5: joint_condition_proxy（仅在接近 dock 时才有意义） ----
    # 关键修复：pos_factor 用陡峭衰减，远离 dock 时接近 0，避免"活着刷分"
    # 门控：只有 nxt_dist < 2.0 时才计算，且权重大幅降低
    if nxt_dist < 2.0:
        pos_factor = 1.0 / (1.0 + 10.0 * nxt_dist)
        align_factor = align_cos
        vel_factor = 1.0 / (1.0 + 20.0 * nxt_crate_speed)
        joint_proxy = (pos_factor * align_factor * vel_factor) ** (1.0 / 3.0)
        joint_reward = 1.0 * joint_proxy
    else:
        joint_reward = 0.0

    # ---- 组件 6: impact_softness ----
    impact_penalty = 0.0
    if contact > 0.5:
        crate_speed_jump = nxt_crate_speed - cur_crate_speed
        if crate_speed_jump > 0.0:
            impact_penalty -= 0.5 * crate_speed_jump
        if nxt_crate_speed > 1.0:
            impact_penalty -= 0.3 * (nxt_crate_speed - 1.0)

    # ---- 组件 7: boundary_avoidance ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    boundary_penalty = 0.0
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    if front_prox > 0.8:
        boundary_penalty -= 0.3 * (front_prox - 0.8)

    cart_cos = next_obs[2]
    cart_sin = next_obs[3]
    rel_bx = next_obs[6] * 3.0
    rel_by = next_obs[7] * 3.0
    crate_wx = cart_cos * rel_bx - cart_sin * rel_by + cart_x * 5.0
    crate_wy = cart_sin * rel_bx + cart_cos * rel_by + cart_y * 4.0
    if abs(crate_wx) > 4.3:
        boundary_penalty -= 0.5 * (abs(crate_wx) - 4.3)
    if abs(crate_wy) > 3.4:
        boundary_penalty -= 0.5 * (abs(crate_wy) - 3.4)

    # ---- 组件 8: action_smoothness ----
    drive = action[0]
    steer = action[1]
    action_penalty = -0.02 * (drive * drive + steer * steer)

    # ---- 总奖励 ----
    components = {}
    components["crate_to_dock_progress"] = float(progress_reward)
    components["contact_gated_push"] = float(push_reward)
    components["crate_dock_alignment"] = float(alignment_reward)
    components["crate_settling"] = float(settling_reward)
    components["joint_condition_proxy"] = float(joint_reward)
    components["impact_softness"] = float(impact_penalty)
    components["boundary_avoidance"] = float(boundary_penalty)
    components["action_smoothness"] = float(action_penalty)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components
```
