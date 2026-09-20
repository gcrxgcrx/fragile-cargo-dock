# Response Record

分析：任务分数 3.71 且 20/20 全部 truncation、无成功终止，说明策略从未真正完成停靠——它在 dock 附近"悬停收割" `dock_proxy_reward`（占总量 ~90%，激活率 63%），而真正的主进度信号 `crate_to_dock_progress` 只有 ~28 且几乎不再增长。`dock_proxy_reward` 的 `still_factor` 是持续全局奖励，且门控只要求"接近+本步靠近"，导致货箱停在附近即可长期收分，与"必须低速对齐停稳"脱节。改进方向：把 dock 奖励改成**只在真正接近且正在靠近时、以联合条件（近+对齐+低速）为乘性门控的稀疏信号**，并大幅降低其权重；强化 delta 主信号；把低速/对齐改为门控而非独立收分项；对齐项改为双向惩罚（当前只奖励 cos 增大，负向几乎不罚）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 观测解包 ----------
    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0
    next_crate_vx = next_obs[8] * 3.0
    next_crate_vy = next_obs[9] * 3.0

    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    next_dx = next_obs[12] * 5.0
    next_dy = next_obs[13] * 4.0

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (next_dx * next_dx + next_dy * next_dy) ** 0.5

    # 货箱朝向对齐度（假设 dock 朝向为世界 x 轴）
    cos_h = obs[10]
    next_cos_h = next_obs[10]
    align = 0.5 * (cos_h + 1.0)
    next_align = 0.5 * (next_cos_h + 1.0)

    speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    next_speed = (next_crate_vx * next_crate_vx + next_crate_vy * next_crate_vy) ** 0.5

    contact = obs[14]
    cart_speed = obs[4] * 3.0
    sensor_front = obs[15]
    sensor_left = obs[16]
    sensor_right = obs[17]

    # ---------- 组件 1：货箱向 dock 的进度（delta 距离，主信号，凸化） ----------
    progress_raw = (dist - next_dist)
    if progress_raw > 0.5:
        progress_raw = 0.5
    if progress_raw < -0.5:
        progress_raw = -0.5
    # 凸化：正向靠近给更强梯度，负向退步按线性惩罚
    if progress_raw > 0.0:
        progress = 14.0 * progress_raw
    else:
        progress = 10.0 * progress_raw

    # ---------- 组件 2：朝向对齐（双向误差惩罚，避免悬停不罚） ----------
    # 对齐误差：1 - align，范围 [0,1]，越小越好
    align_err = 1.0 - next_align
    align_penalty = -0.6 * (align_err * align_err)

    # ---------- 组件 3：停靠联合条件（稀疏门控，仅在接近+靠近+低速+对齐时给小额奖励） ----------
    near_factor = 1.0 / (1.0 + 3.0 * next_dist)          # 接近因子
    align_factor = next_align                             # 对齐因子
    still_factor = 1.0 / (1.0 + 20.0 * next_speed)        # 低速因子（仅作门控）
    dock_proxy = (near_factor * align_factor * still_factor) ** (1.0 / 3.0)
    dock_reward = 0.0
    # 门控：真正接近 dock、本步在靠近、且货箱已低速，才给奖励
    if next_dist < 0.8 and progress_raw > 0.0 and next_speed < 0.6:
        dock_reward = 0.6 * dock_proxy

    # ---------- 组件 4：轻柔接触（hinge，仅接触且相对速度高时惩罚） ----------
    gentle_penalty = 0.0
    if contact > 0.5:
        rel_speed = (cart_speed * cart_speed + next_speed * next_speed) ** 0.5
        excess = rel_speed - 1.5
        if excess > 0.0:
            gentle_penalty = -0.3 * excess

    # ---------- 组件 5：边界规避（hinge） ----------
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    if cart_x > 0.75:
        boundary_penalty -= 0.2 * (cart_x - 0.75) / 0.25
    if cart_x < -0.75:
        boundary_penalty -= 0.2 * (-0.75 - cart_x) / 0.25
    if cart_y > 0.75:
        boundary_penalty -= 0.2 * (cart_y - 0.75) / 0.25
    if cart_y < -0.75:
        boundary_penalty -= 0.2 * (-0.75 - cart_y) / 0.25
    if next_dist > 4.0:
        boundary_penalty -= 0.1 * (next_dist - 4.0)

    # ---------- 组件 6：静态障碍接近惩罚（hinge） ----------
    obstacle_penalty = 0.0
    if sensor_front > 0.7:
        obstacle_penalty -= 0.15 * (sensor_front - 0.7) / 0.3
    if sensor_left > 0.8:
        obstacle_penalty -= 0.1 * (sensor_left - 0.8) / 0.2
    if sensor_right > 0.8:
        obstacle_penalty -= 0.1 * (sensor_right - 0.8) / 0.2

    # ---------- 组件 7：动作平滑（轻量） ----------
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {}
    components["crate_to_dock_progress"] = progress
    components["crate_dock_alignment"] = align_penalty
    components["dock_proxy_reward"] = dock_reward
    components["gentle_contact_penalty"] = gentle_penalty
    components["boundary_penalty"] = boundary_penalty
    components["obstacle_penalty"] = obstacle_penalty
    components["action_smoothness_penalty"] = smooth_penalty

    total_reward = (
        components["crate_to_dock_progress"]
        + components["crate_dock_alignment"]
        + components["dock_proxy_reward"]
        + components["gentle_contact_penalty"]
        + components["boundary_penalty"]
        + components["obstacle_penalty"]
        + components["action_smoothness_penalty"]
    )

    return float(total_reward), components
```
