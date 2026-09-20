# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到泊位的有符号偏移（已归一化）
    dx_now = obs[12]
    dy_now = obs[13]
    dx_next = next_obs[12]
    dy_next = next_obs[13]

    dist_now = (dx_now * dx_now + dy_now * dy_now) ** 0.5
    dist_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5

    # 泊位进入阈值（由卡片给出）
    in_x = 0.024
    in_y = 0.030
    inside_now = 1.0 if (abs(dx_now) <= in_x and abs(dy_now) <= in_y) else 0.0

    # 货箱速度（世界系，归一化）
    cvx = next_obs[8]
    cvy = next_obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    heading_err = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if heading_err < 1e-6:
        angle_err = 0.0
    else:
        cos_a = next_obs[10] / heading_err
        if cos_a > 1.0:
            cos_a = 1.0
        if cos_a < -1.0:
            cos_a = -1.0
        angle_err = (1.0 - cos_a)  # 0 表示完全对齐，2 表示反向

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    front = next_obs[15]
    left = next_obs[16]
    right = next_obs[17]

    # 小车位置（归一化，仓库半宽/半高）
    cart_x = next_obs[0]
    cart_y = next_obs[1]

    # ---------- 主信号 1: 货箱向泊位推进（delta 形式，避免悬停陷阱） ----------
    progress = (dist_now - dist_next) * 12.0
    if progress > 0.5:
        progress = 0.5
    if progress < -0.5:
        progress = -0.5

    # ---------- 主信号 2: 泊位内质量（位置 + 朝向 + 静止 的联合代理） ----------
    # 位置因子：越接近泊位中心越高，仅在接近泊位时显著
    pos_factor = 1.0 / (1.0 + 40.0 * dist_next)
    # 朝向因子：误差越小越高
    align_factor = 1.0 / (1.0 + 6.0 * angle_err)
    # 静止因子：速度越小越高
    speed_factor = 1.0 / (1.0 + 60.0 * crate_speed)
    # 几何平均，避免乘积塌缩
    dock_quality = (pos_factor * align_factor * speed_factor) ** (1.0 / 3.0)
    dock_quality_bonus = 2.0 * dock_quality * inside_now

    # ---------- 主信号 3: 进入泊位的稀疏 bonus ----------
    entry_bonus = 0.0
    if inside_now > 0.5 and abs(dx_next) <= in_x and abs(dy_next) <= in_y:
        # 仅当从外面进入时给一次（用 obs 判断）
        was_inside = 1.0 if (abs(dx_now) <= in_x and abs(dy_now) <= in_y) else 0.0
        if was_inside < 0.5:
            entry_bonus = 1.5

    # ---------- 次级: 接近泊位时的速度抑制（仅在接近泊位时启用） ----------
    near_dock = 1.0 if dist_next < 0.15 else 0.0
    speed_penalty = -0.6 * near_dock * crate_speed

    # ---------- 次级: 软接触惩罚（仅在接触且速度突变大时轻罚） ----------
    cart_fwd_speed = abs(next_obs[4])
    # 货箱速度与小车速度不一致且接触，说明可能硬碰撞
    soft_contact_penalty = 0.0
    if contact > 0.5:
        mismatch = abs(crate_speed - cart_fwd_speed)
        if mismatch > 0.3:
            soft_contact_penalty = -0.3 * (mismatch - 0.3)

    # ---------- 次级: 越界惩罚（hinge 形式，仅在接近边界时） ----------
    oob_penalty = 0.0
    cart_bound = 0.95
    if abs(cart_x) > cart_bound:
        oob_penalty += -1.0 * (abs(cart_x) - cart_bound)
    if abs(cart_y) > cart_bound:
        oob_penalty += -1.0 * (abs(cart_y) - cart_bound)

    # ---------- 次级: 障碍接近惩罚（hinge） ----------
    obstacle_penalty = 0.0
    if front > 0.85:
        obstacle_penalty += -0.5 * (front - 0.85)
    if left > 0.9:
        obstacle_penalty += -0.3 * (left - 0.9)
    if right > 0.9:
        obstacle_penalty += -0.3 * (right - 0.9)

    # ---------- 次级: 动作平滑（轻量，不压制推动） ----------
    smooth_penalty = -0.02 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 组装 ----------
    components = {}
    components["crate_to_dock_progress"] = float(progress)
    components["dock_quality_bonus"] = float(dock_quality_bonus)
    components["dock_entry_bonus"] = float(entry_bonus)
    components["near_dock_speed_penalty"] = float(speed_penalty)
    components["soft_contact_penalty"] = float(soft_contact_penalty)
    components["out_of_bounds_penalty"] = float(oob_penalty)
    components["obstacle_proximity_penalty"] = float(obstacle_penalty)
    components["action_smoothness_penalty"] = float(smooth_penalty)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
