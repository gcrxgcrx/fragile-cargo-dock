# Response Record

分析：任务分数 -67.4 且训练 episode return 长期停在 -8 左右、几乎不随进度改善，说明当前奖励已被"什么都不做"的稳态锁定——`crate_docking_quality` 的几何平均含 speed_factor，货箱一有速度该因子就塌到 0，等于惩罚一切推动行为；`crate_speed_penalty_near_dock` 与 `soft_contact_penalty` 也在接触推动时直接扣分。同时 `crate_to_dock_progress` 用 delta 距离，货箱静止时恒为 0，无法产生正向梯度。改法：把主进度信号改成"基于货箱到泊位距离的稠密接近奖励 + 每步改进量"的凸化形式，保证推箱必然得分；速度/朝向/静止只作为接近泊位后的门控乘子（不独立扣分）；软接触惩罚只在接触且货箱速度远超推动所需时以 hinge 形式轻微生效，确保推箱净正收益。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取 ----
    crate_dock_x = next_obs[12]
    crate_dock_y = next_obs[13]
    prev_dock_x = obs[12]
    prev_dock_y = obs[13]

    crate_vx = next_obs[8]
    crate_vy = next_obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    # 朝向误差（弧度），0 表示对齐
    heading_error = (crate_sin * crate_sin + (1.0 - crate_cos) * (1.0 - crate_cos)) ** 0.5

    contact = next_obs[14]
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    prev_dist = (prev_dock_x * prev_dock_x + prev_dock_y * prev_dock_y) ** 0.5
    curr_dist = (crate_dock_x * crate_dock_x + crate_dock_y * crate_dock_y) ** 0.5

    # ---- 1. 主进度：稠密接近奖励（凸化）+ 每步改进量 ----
    # 接近奖励：距离越近越高，静止在初始位置时该值小且恒定，推近必然增大
    proximity = max(0.0, 1.0 - curr_dist / 0.6)
    approach_reward = 4.0 * proximity * proximity
    # 改进量：鼓励每步缩短距离（对推箱行为恒为正贡献）
    improvement = prev_dist - curr_dist
    progress_reward = 3.0 * improvement
    crate_to_dock_progress = approach_reward + progress_reward

    # ---- 2. 停靠质量门控（乘子，不作为独立扣分）----
    # 位置因子
    pos_x_factor = max(0.0, 1.0 - abs(crate_dock_x) / 0.05)
    pos_y_factor = max(0.0, 1.0 - abs(crate_dock_y) / 0.06)
    pos_factor = pos_x_factor * pos_y_factor
    # 朝向因子
    heading_factor = max(0.0, 1.0 - heading_error / 0.5236)
    # 静止因子（仅在已进入泊位附近才生效，避免惩罚推动）
    near_dock = max(0.0, 1.0 - curr_dist / 0.15)
    speed_factor = max(0.0, 1.0 - crate_speed / 0.04)

    # 联合质量：几何平均，连续可导
    docking_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    # 停靠质量奖励：仅在接近泊位时给正奖励，鼓励最终满足复合条件
    crate_docking_quality = 5.0 * near_dock * docking_quality

    # ---- 3. 越界惩罚（hinge，仅边界附近）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_edge = max(abs(cart_x), abs(cart_y))
    out_of_bounds_penalty = -2.0 * max(0.0, cart_edge - 0.9)

    # ---- 4. 障碍接近惩罚（hinge，仅在非常接近时）----
    obstacle_penalty = -0.3 * (max(0.0, sensor_front - 0.9) +
                               max(0.0, sensor_left - 0.9) +
                               max(0.0, sensor_right - 0.9))

    # ---- 5. 软接触惩罚：仅当接触且货箱速度远超推动所需时轻微抑制 ----
    # 正常推动所需速度远低于此阈值，故不会惩罚任务进展本身
    if contact > 0.5:
        excess = max(0.0, crate_speed - 0.15)
        soft_contact_penalty = -0.5 * excess
    else:
        soft_contact_penalty = 0.0

    # ---- 6. 动作平滑（极轻量）----
    action_smoothness = -0.02 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_docking_quality": float(crate_docking_quality),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = sum(components.values())
    return (float(total_reward), components)
```
