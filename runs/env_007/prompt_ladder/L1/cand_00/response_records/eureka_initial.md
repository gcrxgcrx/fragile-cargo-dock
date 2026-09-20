# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取（仅使用已声明维度）----
    # 货箱到泊位的有符号偏移（归一化）
    dx = next_obs[12]
    dy = next_obs[13]
    odx = obs[12]
    ody = obs[13]

    # 货箱世界系速度（归一化），恢复为米/秒量级
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向误差（弧度）
    crate_heading = (next_obs[11] ** 2 + next_obs[10] ** 2) ** 0.5
    if crate_heading < 1e-6:
        heading_err = 0.0
    else:
        heading_err = (next_obs[11] ** 2) ** 0.5
    # 用 cos 值更稳健地表达对齐程度
    cos_align = next_obs[10]
    if cos_align > 1.0:
        cos_align = 1.0
    if cos_align < -1.0:
        cos_align = -1.0

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    front = next_obs[15]
    left = next_obs[16]
    right = next_obs[17]

    # 小车位置（归一化）
    cx = next_obs[0]
    cy = next_obs[1]

    # ---- 组件 1：货箱向泊位的进度（delta 形式，避免悬停）----
    old_dist = (odx * odx + ody * ody) ** 0.5
    new_dist = (dx * dx + dy * dy) ** 0.5
    progress = old_dist - new_dist
    # 压缩到合理范围
    progress = progress / (1.0 + abs(progress))
    crate_progress = 8.0 * progress

    # ---- 组件 2：泊位接近度（bounded，仅在靠近时提供引导）----
    near_gate = 1.0 / (1.0 + 6.0 * new_dist)
    dock_proximity = 1.5 * near_gate

    # ---- 组件 3：停靠质量（联合条件 proxy：进入 + 对齐 + 静止）----
    # 进入条件：偏移越小越好
    inside_factor = 1.0 / (1.0 + 30.0 * new_dist)
    # 朝向对齐因子：cos 从 -1..1 映射到 0..1
    align_factor = (cos_align + 1.0) * 0.5
    # 静止因子：速度越小越好
    still_factor = 1.0 / (1.0 + 8.0 * crate_speed)
    # 几何平均，避免塌缩
    joint = (inside_factor * align_factor * still_factor) ** (1.0 / 3.0)
    docking_quality = 4.0 * joint * near_gate

    # ---- 组件 4：接近泊位时的速度抑制（门控，仅在靠近时启用）----
    speed_penalty = -1.0 * near_gate * (crate_speed ** 2)

    # ---- 组件 5：轻柔接触（避免硬碰撞的间接推断）----
    # 接触时若货箱速度突变过大则惩罚；这里用接触 * 速度平方作为温和抑制
    if contact > 0.5:
        soft_contact = -0.5 * (crate_speed ** 2)
    else:
        soft_contact = 0.0

    # ---- 组件 6：越界惩罚（hinge，仅接近边界时生效）----
    bound_penalty = 0.0
    if abs(cx) > 0.9:
        bound_penalty = bound_penalty - 2.0 * (abs(cx) - 0.9) ** 2
    if abs(cy) > 0.9:
        bound_penalty = bound_penalty - 2.0 * (abs(cy) - 0.9) ** 2

    # ---- 组件 7：障碍接近惩罚（hinge）----
    obstacle_penalty = 0.0
    if front > 0.7:
        obstacle_penalty = obstacle_penalty - 0.5 * (front - 0.7)
    if left > 0.7:
        obstacle_penalty = obstacle_penalty - 0.3 * (left - 0.7)
    if right > 0.7:
        obstacle_penalty = obstacle_penalty - 0.3 * (right - 0.7)

    # ---- 组件 8：动作平滑（轻量）----
    action_penalty = -0.05 * (action[0] ** 2 + action[1] ** 2)

    components = {
        "crate_progress": crate_progress,
        "dock_proximity": dock_proximity,
        "docking_quality": docking_quality,
        "speed_penalty_near_dock": speed_penalty,
        "soft_contact": soft_contact,
        "bound_penalty": bound_penalty,
        "obstacle_penalty": obstacle_penalty,
        "action_penalty": action_penalty,
    }

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]

    return (float(total_reward), components)
```
