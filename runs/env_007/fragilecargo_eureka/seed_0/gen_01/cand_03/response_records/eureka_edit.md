# Response Record

分析：任务分数为负、20/20 全部 truncation，说明策略从未真正完成 dock 停靠；而 `joint_completion_proxy` 每步几乎恒为 0.7 左右（episode 累计 215），它只依赖“货箱离 dock 近 + 朝向尚可 + 速度不高”，在货箱静止于任意位置时也能刷满，完全主导了总奖励（占比 ~89%），使策略学会“不动/慢速悬停”而非推进。同时 `crate_to_dock_progress` 激活率仅 ~1%、`contact_gated_push` 几乎为 0，真正的推动信号被淹没。改进方向：把 joint proxy 改为**必须接近 dock 才显著**的强门控形式并大幅降权，凸化并提高进度 delta 权重，把 alignment/settling 也改为只在 near-dock 时生效，避免“远处静止”刷分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 尺度常量 ----
    # 货箱到 dock 距离（米）：obs[12]*5.0, obs[13]*4.0
    # 货箱速率（m/s）：obs[8]*3.0, obs[9]*3.0
    # 朝向误差幅度近似：|sin(theta_err)| = |obs[11]| / norm

    def crate_dock_dist(o):
        dx = o[12] * 5.0
        dy = o[13] * 4.0
        return (dx * dx + dy * dy) ** 0.5

    def crate_speed(o):
        vx = o[8] * 3.0
        vy = o[9] * 3.0
        return (vx * vx + vy * vy) ** 0.5

    def crate_align_err(o):
        norm = (o[10] * o[10] + o[11] * o[11]) ** 0.5
        if norm < 1.0e-6:
            return 1.0
        return abs(o[11] / norm)

    # ---- 状态量 ----
    d_prev = crate_dock_dist(obs)
    d_next = crate_dock_dist(next_obs)
    spd_prev = crate_speed(obs)
    spd_next = crate_speed(next_obs)
    align_err_next = crate_align_err(next_obs)

    contact_next = next_obs[14]

    # near_gate：仅在货箱接近 dock 时才显著（0 远，1 在中心）
    # 用较陡的衰减，避免远处静止也能拿分
    near_gate = 1.0 / (1.0 + 4.0 * d_next)

    # ---- 1. 主进度：货箱向 dock 靠近（delta 形式，凸化放大有效推进） ----
    progress = d_prev - d_next
    if progress > 0.0:
        crate_to_dock_progress = 25.0 * progress
    else:
        crate_to_dock_progress = 12.0 * progress  # 远离时轻罚，防止倒退

    # ---- 2. 朝向对齐：仅在接近 dock 时启用 ----
    align_reward = near_gate * max(0.0, 1.0 - align_err_next / 0.5)
    crate_dock_alignment = 2.0 * align_reward

    # ---- 3. 停稳：接近 dock 时抑制货箱速度（hinge） ----
    settle_penalty = near_gate * max(0.0, spd_next - 0.05)
    crate_settling = -6.0 * settle_penalty

    # ---- 4. 接触门控推动：接触且货箱在动时给予正信号 ----
    push_signal = 0.0
    if contact_next > 0.5:
        push_signal = min(1.0, spd_next / 1.0)
    contact_gated_push = 1.5 * push_signal

    # ---- 5. 冲击柔化：接触时速度突变过大则惩罚 ----
    impact_penalty = 0.0
    if contact_next > 0.5:
        dv = abs(spd_next - spd_prev)
        impact_penalty = max(0.0, dv - 0.5)
    impact_softness = -2.0 * impact_penalty

    # ---- 6. 边界规避：小车越界风险（hinge），障碍接近度辅助 ----
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    boundary_penalty += max(0.0, abs(cart_x) - 0.85)
    boundary_penalty += max(0.0, abs(cart_y) - 0.85)
    obstacle_closeness = max(obs[15], obs[16], obs[17])
    boundary_penalty += 0.3 * max(0.0, obstacle_closeness - 0.9)
    boundary_avoidance = -4.0 * boundary_penalty

    # ---- 7. 联合完成近似：必须“接近 dock”才显著，避免远处静止刷分 ----
    # 用 near_gate 强门控，使远处静止时该分量趋近 0
    f_near = 1.0 / (1.0 + 6.0 * d_next)
    f_align = max(0.0, 1.0 - align_err_next / 0.5)
    f_still = max(0.0, 1.0 - spd_next / 0.5)
    joint_proxy = (f_near * f_align * f_still) ** (1.0 / 3.0)
    # 强门控：只有货箱真的靠近 dock 时才有意义
    joint_completion_proxy = 3.0 * (near_gate ** 2) * joint_proxy

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_dock_alignment": float(crate_dock_alignment),
        "crate_settling": float(crate_settling),
        "contact_gated_push": float(contact_gated_push),
        "impact_softness": float(impact_softness),
        "boundary_avoidance": float(boundary_avoidance),
        "joint_completion_proxy": float(joint_completion_proxy),
    }

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return (float(total_reward), components)
```
