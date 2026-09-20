`evidence`: 上一轮 score=17.69（best），19/20 truncated、1/20 terminated，len≈392 说明 agent 长期存活但几乎不完成交付；组件统计显示 `dock_gate` episode_sum_mean=233.3、magnitude_share=95.3%、active_rate=71%，而真正推进的 `crate_to_dock_progress` 仅 11.35（4.6%），即门控乘子被当成独立持续收益刷分，主信号被淹没。

`behavior_diagnosis`: 策略学会了"贴着坞附近、对齐、慢速"的静止稳态——门控值高就持续拿分，但货箱并未被真正推入坞内满足 success 条件（朝向<30°、速度<0.05、持续10步），因此 19/20 超时截断。

`signal_completeness`: 交付所需职责（推进、进坞、朝向对齐、近静止、持续）在 obs 中全部可观测，但当前骨架把"接近+对齐+慢"做成独立乘子收益，导致职责被错误地当成可单独刷分的状态值，缺"联合完成"的收敛信号。

`selected_level`: Level 3 重建（累积记录同骨架族连续 4 轮未刷新 best，且历史最佳 17.69 远低于 target×0.5=125）。

`selected_intervention`: 更换主信号框架——废弃"门控乘子独立收分"，改为 `improvement_delta`（货箱到坞距离改善）+ `joint_condition_proxy`（进坞/对齐/近静止三因子几何平均，仅在接近坞时激活）作为唯一主信号，辅助仅保留 hinge 边界与接触冲击。

`falsifiable_hypothesis`: 若把收益从"状态值"改为"改善量 + 联合完成度"，则静止贴坞不再积累收益，agent 必须持续减小 dist 并同时满足对齐/静止才能得分，从而提升 docked_success 触发率与 score。

`expected_next_round`: score 应 > 17.69（目标 >30）；`crate_to_dock_progress` magnitude_share 应升至 >40%；`dock_gate` 类独立状态收益消失（新组件 `joint_completion` active_rate 应 >20% 且 episode_sum_mean 不再主导）；terminated 中 success-like 比例上升，len 分布出现 <392 的成功回合。

`main_risk`: 几何平均因子若在早期全为 0 会导致信号稀疏；已用下限 0.15 与"接近坞才激活"缓解，但若 agent 长期无法接近坞，`joint_completion` 可能长期为 0，退化为纯 progress 信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱世界系速率 (m/s)
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 货箱到坞距离的改善量（唯一推进信号） ----------
    progress = dist - next_dist  # 靠近为正
    progress_reward = 12.0 * progress

    # ---------- 主信号 B: 联合完成度（进坞 + 对齐 + 近静止），仅在接近坞时激活 ----------
    # 进坞因子：dist 越小越接近 1（终止边界约 0.15，阈值 0.5 给足缓冲）
    dock_factor = max(0.0, 1.0 - dist / 0.5)
    # 对齐因子：朝向误差越小越接近 1（下限 0.15 防塌缩）
    align_gate = max(0.15, align_factor)
    # 静止因子：货箱越慢越接近 1（下限 0.15 防塌缩）
    slow_gate = max(0.15, 1.0 - crate_speed / 0.5)
    # 几何平均，避免乘积塌缩
    joint = (dock_factor * align_gate * slow_gate) ** (1.0 / 3.0)
    # 只在接近坞时激活联合完成信号，避免远处刷分
    joint_reward = 3.0 * joint * dock_factor

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85 = 终止边界 1.0 的 85%） ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 组件 D: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + joint_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "joint_completion": float(joint_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```