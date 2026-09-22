# 设计理由
本轮进行 **Level 2 结构变换**，修改 `landing_reward` 组件：将其中的 `contact_factor` 从 `min(nl_contact, nr_contact)` 改为 `(nl_contact + nr_contact) / 2.0`。

**原因**：当前代理快速失败（平均 68 步，全 terminate），`landing_reward` 因要求双腿同时触地（`min` = 0 或 1）而在多数时刻为 0，没有梯度可引导着陆过程。改为连续化后，即使只有单腿触地也能获得部分正向奖励，形成稠密梯度，辅助代理学习稳定着陆；同时保持双腿触地时的最大奖励。系数不变（仍 0.05），避免过度拉动总分。

该修改遵循“乘积 proxy 经常塌缩为 0 → 连续 bounded factor”原则，且不新增独立惩罚，不影响已稳定的负反馈平衡。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, l_contact, r_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, n_angle, n_ang_vel, nl_contact, nr_contact = next_obs

    current_dist = (x_pos**2 + y_pos**2) ** 0.5
    next_dist    = (nx_pos**2 + ny_pos**2) ** 0.5

    current_speed = (x_vel**2 + y_vel**2) ** 0.5
    next_speed    = (nx_vel**2 + ny_vel**2) ** 0.5

    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current

    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    if action == 0:
        fuel_cost = 0.0
    elif action == 2:
        fuel_cost = -0.1
    else:
        fuel_cost = -0.05

    crash_risk = max(0.0, -ny_vel - 0.1)
    proximity = max(0.0, 1.0 - next_dist / 0.3)
    crash_prevention = -0.05 * crash_risk * proximity

    # 连续化接触因子：从 min(nl_contact, nr_contact) 改为平均值，提供单腿梯度
    contact_factor = (nl_contact + nr_contact) / 2.0  # 范围 0.0 ~ 1.0
    dist_factor = max(0.0, 1.0 - next_dist / 0.3)
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    landing_reward = 0.05 * contact_factor * dist_factor * speed_factor

    total = progress + orientation_penalty + fuel_cost + crash_prevention + landing_reward

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
        "crash_prevention": crash_prevention,
        "landing_reward": landing_reward
    }
    return float(total), components
```

# 诊断摘要
- **audit**: 缺失水平边界预警，但主要缺口是着陆信号过稀疏（双腿接触要求太严），未给单腿着陆提供梯度。
- **behavior**: 代理快速失败（平均 68 步 crash/飞出），分数 -122 附近，未学会任何着陆策略。
- **signal**: `landing_reward` 因 `contact_factor = min(...)` 二值化而经常为 0，无法形成连续的着陆进度梯度。
- **level**: Level 2
- **hypothesis**: 连续化接触因子后，代理在探索中偶尔单腿触地即能获得正向引导，从而逐渐学会调整姿态完成双腿着陆，提升最终得分并延长生存步数。
- **risk**: 如果代理仅满足于单腿触地的局部奖励而不再追求完全稳定着陆，可能陷入次优策略；但结合 `progress` 等其它信号，整体目标仍会趋向完全着陆。