# Response Record

分析：当前 `dock_proxy_reward` 独占约 90% 的奖励量级，且其"静止因子" `1/(1+10*speed)` 是一个全局持续状态奖励——推动货箱必然产生速度，于是该因子在惩罚任务进展本身，策略被诱导"靠近 dock 后就不动"。同时 `crate_to_dock_progress` 与 `crate_dock_alignment` 量级过小（≈10 与 ≈-0.03），几乎无梯度；`obstacle_penalty` 恒为 0（阈值 0.85 永不触发）；`action_smoothness_penalty` 全时触发但量级小。所有 episode 均 truncation、无 terminated，说明策略学会了"在 dock 附近低速悬停收割 proxy"，而非真正完成停靠。改进方向：把静止/对齐/接近改成**门控因子**乘到"进度 delta"上而非独立持续奖励；用 delta 距离作主信号并放大；把对齐改成 delta 形式；把 proxy 改为仅在接近 dock 且正在改善时生效的小额联合奖励；去掉恒零的 obstacle_penalty 或降低阈值；保留轻柔接触与边界 hinge。

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

    # ---------- 组件 1：货箱向 dock 的进度（delta 距离，主信号） ----------
    progress_raw = (dist - next_dist)
    if progress_raw > 0.5:
        progress_raw = 0.5
    if progress_raw < -0.5:
        progress_raw = -0.5
    progress = progress_raw * 8.0

    # ---------- 组件 2：朝向对齐进度（delta） ----------
    align_delta = (next_align - align) * 1.0

    # ---------- 组件 3：停靠联合条件（门控式，仅接近 dock 且正在改善时给小额奖励） ----------
    # 接近因子（连续，仅在接近时显著）
    near_factor = 1.0 / (1.0 + 2.0 * next_dist)
    # 对齐因子
    align_factor = next_align
    # 静止因子：仅作为门控，不作为全局持续奖励
    still_factor = 1.0 / (1.0 + 10.0 * next_speed)
    # 几何平均避免塌缩
    dock_proxy = (near_factor * align_factor * still_factor) ** (1.0 / 3.0)
    # 门控：只在货箱接近 dock 且本步在靠近时才给奖励，避免悬停收割
    dock_reward = 0.0
    if next_dist < 1.5 and progress_raw > 0.0:
        dock_reward = 0.8 * dock_proxy

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

    # ---------- 组件 6：静态障碍接近惩罚（hinge，阈值下调以真正触发） ----------
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
    components["crate_dock_alignment"] = align_delta
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
