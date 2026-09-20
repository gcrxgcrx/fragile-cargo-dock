# Response Record

分析：任务分数 3.75 且 20/20 全部 truncation，说明策略从未真正完成停靠，只是在 dock 附近"悬停收割"——`dock_proxy_reward`（episode 均值 61.7）和 `crate_to_dock_progress`（44.7）合计贡献了绝大部分正分，而这两者都不要求真正静止对齐完成。`crate_dock_alignment` 几乎全程激活（98.6%）却均值仅 -1.7，说明对齐惩罚太弱、被悬停收益淹没。`action_smoothness_penalty` 全程 100% 激活且量级不小，属于无差别压制。改法：把 dock_proxy 从"接近即给分"改为"仅在真正低速+对齐+接近时给分"的强门控联合条件，并提高其权重以驱动完成；把对齐误差改为 hinge（只在误差大时罚）；把平滑惩罚降权为轻量；进度信号保留但降权，避免悬停收割。

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

    # ---------- 组件 1：货箱向 dock 的进度（delta 距离，降权避免悬停收割） ----------
    progress_raw = (dist - next_dist)
    if progress_raw > 0.5:
        progress_raw = 0.5
    if progress_raw < -0.5:
        progress_raw = -0.5
    if progress_raw > 0.0:
        progress = 6.0 * progress_raw
    else:
        progress = 8.0 * progress_raw

    # ---------- 组件 2：朝向对齐（hinge，仅误差大时惩罚，避免全程压制） ----------
    align_err = 1.0 - next_align
    align_penalty = 0.0
    if align_err > 0.15:
        align_penalty = -1.2 * (align_err - 0.15)

    # ---------- 组件 3：停靠联合条件（强门控，仅在真正接近+低速+对齐时给分） ----------
    # 接近因子：仅在 dist < 0.6 内有效，否则为 0
    near_factor = 0.0
    if next_dist < 0.6:
        near_factor = 1.0 - next_dist / 0.6
    # 低速因子：仅在 speed < 0.3 内有效
    still_factor = 0.0
    if next_speed < 0.3:
        still_factor = 1.0 - next_speed / 0.3
    # 对齐因子：仅在 align > 0.85 内有效
    align_factor = 0.0
    if next_align > 0.85:
        align_factor = (next_align - 0.85) / 0.15
    dock_proxy = (near_factor * still_factor * align_factor) ** (1.0 / 3.0)
    dock_reward = 3.0 * dock_proxy

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
    smooth_penalty = -0.005 * (action[0] * action[0] + action[1] * action[1])

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
