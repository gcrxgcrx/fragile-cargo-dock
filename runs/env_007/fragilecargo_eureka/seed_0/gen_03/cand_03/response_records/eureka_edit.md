# Response Record

分析：训练全程 20/20 都是 truncation，没有任何成功终止，任务分数仅 3.55，说明策略从未真正完成"货箱入 dock + 对齐 + 静止 10 步"。`crate_dock_alignment` 在远离 dock 时也大量激活（92%+），它用的是货箱自身朝向的 |sin|，与 dock 朝向无关，属于"货箱随便转个角度就白拿分"的伪信号，且数值最大（~172/回合），主导了策略。`contact_gated_push` 奖励"接触且货箱在动"，鼓励持续推着货箱跑，与"最终要静止在 dock 内"直接冲突。`joint_completion_proxy` 用 near_gate² 门控但因子过松，远处也能拿分。`impact_softness`、`boundary_avoidance` 几乎不触发（0.1%/0%），形同虚设。改进方向：把 alignment 改为只在接近 dock 时生效并降权；把 push 改为"接触且朝 dock 推进"的 delta 门控；强化 joint proxy 的 near 门控与静止因子；把 settling 与 joint 结合成"入 dock 停稳"的强信号；用 hinge 让 impact/boundary 在真正危险时才罚。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 尺度：obs[12]*5.0, obs[13]*4.0 为货箱到 dock 的米制偏移
    # obs[8]*3.0, obs[9]*3.0 为货箱米制速度

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

    d_prev = crate_dock_dist(obs)
    d_next = crate_dock_dist(next_obs)
    spd_prev = crate_speed(obs)
    spd_next = crate_speed(next_obs)
    align_err_next = crate_align_err(next_obs)

    contact_next = next_obs[14]

    # near_gate：货箱接近 dock 时才显著（0 远，1 在中心）
    near_gate = 1.0 / (1.0 + 6.0 * d_next)

    # ---- 1. 主进度：货箱向 dock 靠近（delta，凸化推进） ----
    progress = d_prev - d_next
    if progress > 0.0:
        crate_to_dock_progress = 30.0 * progress
    else:
        crate_to_dock_progress = 15.0 * progress

    # ---- 2. 朝向对齐：仅当货箱接近 dock 时才启用，并降权 ----
    align_reward = (near_gate ** 2) * max(0.0, 1.0 - align_err_next / 0.5)
    crate_dock_alignment = 1.5 * align_reward

    # ---- 3. 停稳：接近 dock 时抑制货箱速度（hinge） ----
    settle_penalty = (near_gate ** 2) * max(0.0, spd_next - 0.05)
    crate_settling = -8.0 * settle_penalty

    # ---- 4. 接触门控推动：接触且货箱确实朝 dock 靠近时给正信号 ----
    push_signal = 0.0
    if contact_next > 0.5 and progress > 0.0:
        push_signal = min(1.0, progress / 0.02)
    contact_gated_push = 1.0 * push_signal

    # ---- 5. 冲击柔化：接触时速度突变过大则惩罚（hinge） ----
    impact_penalty = 0.0
    if contact_next > 0.5:
        dv = abs(spd_next - spd_prev)
        impact_penalty = max(0.0, dv - 0.4)
    impact_softness = -3.0 * impact_penalty

    # ---- 6. 边界规避：小车越界风险（hinge），障碍接近度辅助 ----
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    boundary_penalty += max(0.0, abs(cart_x) - 0.8)
    boundary_penalty += max(0.0, abs(cart_y) - 0.8)
    obstacle_closeness = max(obs[15], obs[16], obs[17])
    boundary_penalty += 0.3 * max(0.0, obstacle_closeness - 0.85)
    boundary_avoidance = -5.0 * boundary_penalty

    # ---- 7. 联合完成近似：接近 + 对齐 + 静止，强 near 门控 ----
    f_near = 1.0 / (1.0 + 8.0 * d_next)
    f_align = max(0.0, 1.0 - align_err_next / 0.5)
    f_still = max(0.0, 1.0 - spd_next / 0.3)
    joint_proxy = (f_near * f_align * f_still) ** (1.0 / 3.0)
    joint_completion_proxy = 6.0 * (near_gate ** 3) * joint_proxy

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
