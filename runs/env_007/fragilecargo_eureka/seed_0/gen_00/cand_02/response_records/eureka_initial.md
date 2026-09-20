# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主信号：货箱到 dock 中心的距离（世界尺度近似） ----
    def dock_dist(o):
        dx = o[12] * 5.0
        dy = o[13] * 4.0
        return (dx * dx + dy * dy) ** 0.5

    d_prev = dock_dist(obs)
    d_next = dock_dist(next_obs)

    # 进度：距离减少量（有界）
    progress_raw = d_prev - d_next
    progress = progress_raw / (1.0 + abs(progress_raw))

    # 接近度：0..1，越接近 dock 越大
    proximity = 1.0 / (1.0 + 1.0 * d_next)

    # 货箱速率（世界系）
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向误差（相对 dock 对齐方向，dock 对齐假设为 0 朝向）
    heading_err = abs(next_obs[11])
    align_factor = 1.0 / (1.0 + 4.0 * heading_err)

    # 静止因子：速度越低越大，仅在接近 dock 时启用
    near_gate = 1.0 / (1.0 + 2.0 * d_next)
    settle_factor = 1.0 / (1.0 + 20.0 * crate_speed)

    # 接触门控推动
    contact = next_obs[14]
    push_gate = contact * (1.0 - min(1.0, d_next / 3.0))

    # ---- 组件 ----
    components = {}

    # 1) 主进度信号：货箱向 dock 靠近
    components["crate_to_dock_progress"] = 2.0 * progress

    # 2) 接近度塑形（有界，避免悬停陷阱：用 delta 为主，proximity 为辅）
    components["crate_dock_proximity"] = 0.4 * proximity

    # 3) 联合条件代理：接近 + 对齐 + 静止（几何平均，避免塌缩）
    joint = (proximity * align_factor * settle_factor) ** (1.0 / 3.0)
    components["dock_joint_condition"] = 1.5 * joint

    # 4) 朝向对齐（仅在接近 dock 时启用）
    components["crate_dock_alignment"] = 0.6 * near_gate * align_factor

    # 5) 静止塑形（仅在接近 dock 时启用）
    components["crate_settling"] = 0.8 * near_gate * settle_factor

    # 6) 接触门控的有效推动
    components["contact_gated_push"] = 0.5 * push_gate

    # 7) 冲击柔性：接触时速度突变惩罚（用货箱速率与小车速度变化近似）
    if contact > 0.5:
        cart_speed_prev = abs(obs[4]) * 3.0
        cart_speed_next = abs(next_obs[4]) * 3.0
        dv = abs(cart_speed_next - cart_speed_prev)
        impact = max(0.0, dv - 0.8)
        components["impact_softness"] = -0.5 * (impact / (1.0 + impact))
    else:
        components["impact_softness"] = 0.0

    # 8) 边界规避：小车接近仓库边缘
    cart_edge = max(abs(next_obs[0]), abs(next_obs[1]))
    edge_pen = max(0.0, cart_edge - 0.85)
    components["cart_boundary_avoidance"] = -1.0 * edge_pen

    # 9) 货箱越界风险：由相对位置反推货箱世界坐标
    ch = next_obs[2]
    sh = next_obs[3]
    rx = next_obs[6] * 3.0
    ry = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + (ch * rx - sh * ry)
    crate_wy = next_obs[1] * 4.0 + (sh * rx + ch * ry)
    crate_edge = max(abs(crate_wx) / 5.0, abs(crate_wy) / 4.0)
    crate_edge_pen = max(0.0, crate_edge - 0.85)
    components["crate_boundary_avoidance"] = -1.0 * crate_edge_pen

    # 10) 静态障碍接近度（前方为主，避免撞墙）
    obs_front = next_obs[15]
    front_pen = max(0.0, obs_front - 0.7)
    components["obstacle_avoidance"] = -0.6 * front_pen

    # 11) 动作平滑（轻量，避免抖动）
    components["action_smoothness"] = -0.05 * (action[0] ** 2 + action[1] ** 2)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
