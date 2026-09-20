# Response Record

分析：主信号 `crate_progress` 几乎恒为 0（激活率 <3%），说明 delta 形式在每步被噪声淹没，策略完全没有被推向泊位；`crate_docking_quality` 高达 ~171 且 100% 激活，成为唯一主导项——它只奖励"低速+对齐+接近"的静态姿态，与"把货箱推进泊位"无关，导致策略学会原地不动地刷分（`dock_hold_bonus`/`speed_near_dock_penalty` 全程为 0，说明货箱从未接近泊位）。任务分数 -1.86、20/20 truncation，正是"什么都不做"的典型表现。改法：把 `crate_progress` 改成对"接近泊位"的稠密势能信号（用 -距离 的凸化形式），让它成为真正的主信号；把 docking_quality 从"全局持续收分"降级为**仅在接近泊位时生效的门控式加分**，并大幅降低其量级；速度抑制只在门控内生效，绝不在远处惩罚货箱速度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 货箱到泊位偏移（归一化量纲） ----------
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # ---------- 主职责 1：货箱向泊位推进（改进量 + 稠密势能） ----------
    # 改进量：单步靠近泊位给正奖励
    progress = cur_dist - nxt_dist
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5
    crate_progress = 30.0 * progress

    # 稠密势能：距离越近奖励越高（凸化，保证每步都有梯度，避免悬停）
    # 使用 (1 - dist) 的正部平方，dist 归一化后典型范围 0~1
    prox = 1.0 - nxt_dist
    if prox < 0.0:
        prox = 0.0
    if prox > 1.0:
        prox = 1.0
    crate_approach = 4.0 * (prox * prox)

    # ---------- 主职责 2：停靠质量（仅在接近泊位时门控生效） ----------
    # 位置因子：越接近泊位中心越接近 1
    pos_factor = 1.0 / (1.0 + 12.0 * nxt_dist)

    # 朝向因子
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    norm = (crate_cos * crate_cos + crate_sin * crate_sin) ** 0.5
    if norm < 1e-6:
        norm = 1e-6
    crate_cos_n = crate_cos / norm
    align_err = 1.0 - crate_cos_n
    if align_err < 0.0:
        align_err = 0.0
    align_factor = 1.0 / (1.0 + 8.0 * align_err)

    # 静止因子
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    speed_factor = 1.0 / (1.0 + 6.0 * crate_speed)

    # 联合条件（几何平均）
    dock_quality = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)

    # 门控：只有货箱足够接近泊位时，停靠质量才生效
    # 距离 < 0.08 时门接近 1，距离 > 0.25 时门接近 0
    if nxt_dist < 0.08:
        near_gate = 1.0
    elif nxt_dist > 0.25:
        near_gate = 0.0
    else:
        near_gate = (0.25 - nxt_dist) / 0.17
    crate_docking = 2.0 * dock_quality * near_gate

    # ---------- 主职责 2b：进入泊位 + 静止的软加成 ----------
    in_dock = 1.0
    if abs(nxt_dx) > 0.024:
        in_dock = 0.0
    if abs(nxt_dy) > 0.030:
        in_dock = 0.0
    still = 1.0 / (1.0 + 20.0 * crate_speed)
    aligned = 1.0 / (1.0 + 10.0 * align_err)
    dock_hold_bonus = 5.0 * in_dock * still * aligned

    # ---------- 条件职责：接近泊位时抑制货箱速度（防滑过，门控内） ----------
    if nxt_dist < 0.15:
        speed_pen = -2.0 * crate_speed * (0.15 - nxt_dist) / 0.15 * 3.0
    else:
        speed_pen = 0.0

    # ---------- 条件职责：软接触惩罚（间接推断硬碰撞） ----------
    contact = next_obs[14]
    if contact > 0.5:
        prev_vx = obs[8] * 3.0
        prev_vy = obs[9] * 3.0
        prev_speed = (prev_vx * prev_vx + prev_vy * prev_vy) ** 0.5
        dv = crate_speed - prev_speed
        if dv > 0.0:
            soft_contact = -0.5 * dv
        else:
            soft_contact = 0.0
    else:
        soft_contact = 0.0

    # ---------- 条件职责：越界惩罚（hinge） ----------
    out_pen = 0.0
    cart_x = obs[0]
    cart_y = obs[1]
    if abs(cart_x) > 0.9:
        out_pen -= 1.0 * (abs(cart_x) - 0.9) / 0.1
    if abs(cart_y) > 0.9:
        out_pen -= 1.0 * (abs(cart_y) - 0.9) / 0.1
    if cur_dist > 1.2:
        out_pen -= 0.5 * (cur_dist - 1.2)

    # ---------- 动作平滑（轻量） ----------
    act_drive = action[0]
    act_steer = action[1]
    action_smooth = -0.02 * (act_drive * act_drive + act_steer * act_steer)

    # ---------- 障碍接近惩罚 ----------
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]
    sensor_pen = 0.0
    if sensor_front > 0.8:
        sensor_pen -= 0.3 * (sensor_front - 0.8) / 0.2
    if sensor_left > 0.8:
        sensor_pen -= 0.2 * (sensor_left - 0.8) / 0.2
    if sensor_right > 0.8:
        sensor_pen -= 0.2 * (sensor_right - 0.8) / 0.2

    components = {
        "crate_progress": crate_progress,
        "crate_approach": crate_approach,
        "crate_docking_quality": crate_docking,
        "dock_hold_bonus": dock_hold_bonus,
        "speed_near_dock_penalty": speed_pen,
        "soft_contact_penalty": soft_contact,
        "out_of_bounds_penalty": out_pen,
        "action_smoothness": action_smooth,
        "obstacle_proximity_penalty": sensor_pen,
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
