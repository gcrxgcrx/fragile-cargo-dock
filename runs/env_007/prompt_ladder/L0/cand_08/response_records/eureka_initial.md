# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取 (仅使用已声明的 obs 索引) ----
    # 货箱到泊位的有符号偏移 (归一化单位)
    cur_dx = obs[12]
    cur_dy = obs[13]
    nxt_dx = next_obs[12]
    nxt_dy = next_obs[13]

    # 货箱到泊位距离 (归一化单位)
    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 货箱速度 (obs[8], obs[9] 为世界系速度 / 3.0)
    crate_speed = (next_obs[8] * next_obs[8] + next_obs[9] * next_obs[9]) ** 0.5

    # 货箱朝向误差 (obs[10], obs[11] 为 cos/sin)
    heading_err = (next_obs[10] * next_obs[10] + next_obs[11] * next_obs[11]) ** 0.5
    # 朝向对齐程度: cos 越接近 1 越对齐 (假设目标朝向为 0 弧度)
    align = next_obs[10]

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    front_prox = next_obs[15]
    left_prox = next_obs[16]
    right_prox = next_obs[17]

    # 小车位置 (归一化)
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # ---- 主职责 1: 货箱向泊位靠近 (delta 形式, 避免悬停陷阱) ----
    progress = cur_dist - nxt_dist
    # 限制单步 progress 幅度, 防止速度突变刷分
    if progress > 0.05:
        progress = 0.05
    if progress < -0.05:
        progress = -0.05
    crate_to_dock_progress = 5.0 * progress

    # ---- 主职责 2: 泊位停靠质量 (位置 + 朝向 + 速度) ----
    # 泊位内位置因子: 越接近中心越好
    # |dx|<=0.024, |dy|<=0.030 为完全进入条件
    pos_factor_x = max(0.0, 1.0 - abs(nxt_dx) / 0.15)
    pos_factor_y = max(0.0, 1.0 - abs(nxt_dy) / 0.15)
    pos_factor = (pos_factor_x * pos_factor_y) ** 0.5

    # 朝向因子: align 从 -1 到 1, 映射到 0..1
    align_factor = max(0.0, min(1.0, (align + 1.0) * 0.5))

    # 速度因子: 越慢越好, 0.05 为阈值
    speed_factor = max(0.0, 1.0 - crate_speed / 0.3)

    # 联合条件: 几何平均, 避免塌缩
    docking_quality = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    # 仅在货箱接近泊位时才有意义
    if nxt_dist < 0.4:
        crate_docking_quality = 3.0 * docking_quality
    else:
        crate_docking_quality = 0.0

    # ---- 条件职责: 接近泊位时抑制货箱速度 ----
    if nxt_dist < 0.25:
        crate_speed_penalty_near_dock = -2.0 * crate_speed * crate_speed
    else:
        crate_speed_penalty_near_dock = 0.0

    # ---- 条件职责: 软接触惩罚 (间接推断) ----
    # 接触时货箱速度过快视为硬碰撞风险
    if contact > 0.5:
        soft_contact_penalty = -1.5 * crate_speed * crate_speed
    else:
        soft_contact_penalty = 0.0

    # ---- 条件职责: 越界惩罚 (hinge 形式) ----
    # 小车位置越界 (归一化坐标超出 ±1.0 附近)
    cart_bound_penalty = 0.0
    if abs(cart_x) > 0.85:
        cart_bound_penalty += -3.0 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        cart_bound_penalty += -3.0 * (abs(cart_y) - 0.85)

    # 货箱越界: 由 obs[12], obs[13] 推断货箱绝对位置
    # 货箱在仓库坐标系中的位置 = 泊位位置 + 偏移, 泊位假设在远侧
    # 使用货箱到泊位偏移的绝对值作为代理: 偏移过大说明货箱远离泊位区域
    crate_bound_penalty = 0.0
    if abs(nxt_dx) > 1.2:
        crate_bound_penalty += -2.0 * (abs(nxt_dx) - 1.2)
    if abs(nxt_dy) > 1.2:
        crate_bound_penalty += -2.0 * (abs(nxt_dy) - 1.2)

    out_of_bounds_penalty = cart_bound_penalty + crate_bound_penalty

    # ---- 条件职责: 障碍接近惩罚 (hinge) ----
    obstacle_penalty = 0.0
    if front_prox > 0.7:
        obstacle_penalty += -1.0 * (front_prox - 0.7)
    if left_prox > 0.7:
        obstacle_penalty += -0.5 * (left_prox - 0.7)
    if right_prox > 0.7:
        obstacle_penalty += -0.5 * (right_prox - 0.7)

    # ---- 条件职责: 动作平滑 (轻量) ----
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    # ---- 汇总 ----
    components = {
        "crate_to_dock_progress": crate_to_dock_progress,
        "crate_docking_quality": crate_docking_quality,
        "crate_speed_penalty_near_dock": crate_speed_penalty_near_dock,
        "soft_contact_penalty": soft_contact_penalty,
        "out_of_bounds_penalty": out_of_bounds_penalty,
        "obstacle_penalty": obstacle_penalty,
        "action_smoothness": action_smoothness,
    }

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + crate_speed_penalty_near_dock
        + soft_contact_penalty
        + out_of_bounds_penalty
        + obstacle_penalty
        + action_smoothness
    )

    return float(total_reward), components
```
