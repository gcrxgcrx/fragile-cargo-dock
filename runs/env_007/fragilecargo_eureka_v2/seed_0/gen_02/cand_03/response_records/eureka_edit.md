# Response Record

分析：任务分数 3.68 且 20/20 全部截断、无成功终止，说明策略被 `dock_proxy_reward` 主导（episode 均值 ~205，占总量 95%+），它奖励的是"靠近+对齐+低速"的持续状态而非真正把箱子送进 dock，导致悬停在 dock 附近刷分而不完成停靠。`crate_dock_alignment` 几乎恒为负且量级极小（-0.03），`obstacle_penalty` 全程为 0（阈值 0.85 永不触发），`action_smoothness_penalty` 全时激活但只是轻量噪声。改进方向：把 dock_proxy 从"持续状态奖励"改为"仅在货箱真正进入 dock 邻域时的高门槛联合奖励"，并大幅提高其权重以提供完成梯度；把进度信号凸化并保留 delta；对齐改为相对 dock 的绝对对齐奖励（而非 delta）；障碍阈值下调使其可触发；保留轻柔与边界 hinge。

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

    # 货箱朝向对齐（假设 dock 朝向为世界 x 轴）
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

    # ---------- 组件 1：货箱向 dock 的进度（delta 距离，凸化） ----------
    raw_progress = dist - next_dist
    # 凸化：正进度放大，负进度保留
    if raw_progress > 0.0:
        progress = 4.0 * raw_progress
    else:
        progress = 2.0 * raw_progress
    if progress > 0.8:
        progress = 0.8
    if progress < -0.8:
        progress = -0.8

    # ---------- 组件 2：朝向对齐（绝对对齐奖励，门控在接近 dock 时） ----------
    align_reward = 0.0
    if next_dist < 2.0:
        align_reward = 0.4 * next_align

    # ---------- 组件 3：停靠联合条件（近 + 对齐 + 静止），高门槛 ----------
    near_factor = 1.0 / (1.0 + 4.0 * next_dist)
    align_factor = next_align
    still_factor = 1.0 / (1.0 + 20.0 * next_speed)
    dock_proxy = (near_factor * align_factor * still_factor) ** (1.0 / 3.0)
    # 仅在货箱真正接近 dock 中心时给奖励，且随接近程度陡增
    if next_dist < 0.8:
        dock_reward = 6.0 * dock_proxy
    else:
        dock_reward = 0.0

    # ---------- 组件 4：轻柔接触（门控 hinge 惩罚） ----------
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

    # ---------- 组件 6：静态障碍接近惩罚（hinge，阈值下调） ----------
    obstacle_penalty = 0.0
    if sensor_front > 0.6:
        obstacle_penalty -= 0.2 * (sensor_front - 0.6) / 0.4
    if sensor_left > 0.7:
        obstacle_penalty -= 0.15 * (sensor_left - 0.7) / 0.3
    if sensor_right > 0.7:
        obstacle_penalty -= 0.15 * (sensor_right - 0.7) / 0.3

    # ---------- 组件 7：动作平滑（轻量） ----------
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 汇总 ----------
    components = {}
    components["crate_to_dock_progress"] = progress
    components["crate_dock_alignment"] = align_reward
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
