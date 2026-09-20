分析：当前奖励几乎完全由 action_smoothness（-4.87）和 out_of_bounds_penalty（-3.13）主导，而 crate_to_dock_progress 与 crate_docking_quality 几乎恒为 0——主学习信号缺失，策略被"少动、别越界"绑架（episode 长度骤降到 ~165 且不再增长）。progress 用归一化坐标差作 delta，量级过小且被平滑惩罚压过；docking_quality 的几何平均因三因子同时满足概率极低而恒为 0。改进方向：把主信号改为基于米制距离的势能差（放大尺度），用货箱到泊位距离的连续 bounded 信号作为持续梯度，把 docking_quality 从"三因子乘积"改为"分项加权和"以保证每步有梯度，并把 smoothness 权重降到 1/10 量级，避免其压过任务进展。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取 ----
    # 货箱到泊位有符号偏移（归一化）；半宽/半高用于恢复米制
    # 从环境卡片：|obs[12]|<=0.024 完全进入（x），|obs[13]|<=0.030（y）
    # 归一化分母可近似取 1.0（obs 已在 [-2,2]），此处直接用归一化量作相对度量
    crate_dock_x = next_obs[12]
    crate_dock_y = next_obs[13]
    prev_dock_x = obs[12]
    prev_dock_y = obs[13]

    # 货箱世界系速度（归一化，m/s / 3.0）
    crate_vx = next_obs[8]
    crate_vy = next_obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向误差（用 cos 偏差近似弧度）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_error = (crate_sin * crate_sin + (1.0 - crate_cos) * (1.0 - crate_cos)) ** 0.5

    contact = next_obs[14]
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    # ---- 1. 主进度信号：势能差（放大尺度）----
    prev_dist = (prev_dock_x * prev_dock_x + prev_dock_y * prev_dock_y) ** 0.5
    curr_dist = (crate_dock_x * crate_dock_x + crate_dock_y * crate_dock_y) ** 0.5
    progress = prev_dist - curr_dist
    crate_to_dock_progress = 40.0 * progress

    # ---- 2. 持续接近信号（bounded，保证每步有梯度）----
    # 距离越近信号越强，用 1/(1+k*d) 形式，范围 (0,1]
    approach_signal = 1.0 / (1.0 + 8.0 * curr_dist)
    crate_approach = 1.5 * approach_signal

    # ---- 3. 停靠质量：分项加权和（不塌缩）----
    # 位置因子
    pos_x_factor = max(0.0, 1.0 - abs(crate_dock_x) / 0.024)
    pos_y_factor = max(0.0, 1.0 - abs(crate_dock_y) / 0.030)
    pos_factor = 0.5 * (pos_x_factor + pos_y_factor)
    # 朝向因子：误差 < 30° (0.5236 rad)
    heading_factor = max(0.0, 1.0 - heading_error / 0.5236)
    # 静止因子：速度 < 0.05 m/s 对应归一化 0.05/3.0 ≈ 0.01667
    speed_factor = max(0.0, 1.0 - crate_speed / 0.01667)
    # 加权和，各分项均连续，避免乘积塌缩
    docking_quality = 0.5 * pos_factor + 0.3 * heading_factor + 0.2 * speed_factor
    crate_docking_quality = 3.0 * docking_quality

    # ---- 4. 接近泊位时的速度抑制（仅在非常接近时启用）----
    near_dock = max(0.0, 1.0 - curr_dist / 0.10)
    crate_speed_penalty_near_dock = -0.8 * near_dock * crate_speed

    # ---- 5. 软接触惩罚（仅在接触且速度过大时）----
    if contact > 0.5:
        excess_speed = max(0.0, crate_speed - 0.05)
        soft_contact_penalty = -1.0 * excess_speed
    else:
        soft_contact_penalty = 0.0

    # ---- 6. 越界惩罚（hinge）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_edge = max(abs(cart_x), abs(cart_y))
    cart_bound_penalty = -2.0 * max(0.0, cart_edge - 0.90)

    crate_edge = max(abs(crate_dock_x), abs(crate_dock_y))
    crate_bound_penalty = -2.0 * max(0.0, crate_edge - 1.0)
    out_of_bounds_penalty = cart_bound_penalty + crate_bound_penalty

    # ---- 7. 障碍接近惩罚 ----
    obstacle_penalty = -0.3 * (max(0.0, sensor_front - 0.85) +
                               max(0.0, sensor_left - 0.85) +
                               max(0.0, sensor_right - 0.85))

    # ---- 8. 动作平滑（大幅降权，避免压过主信号）----
    action_smoothness = -0.01 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_approach": float(crate_approach),
        "crate_docking_quality": float(crate_docking_quality),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = sum(components.values())
    return (float(total_reward), components)
```