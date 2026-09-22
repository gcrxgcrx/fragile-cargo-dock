# Response Record

# 设计理由
当前奖励函数缺少指向成功着陆的明确激励信号，仅依靠势函数 shaping（progress）间接引导，且缺乏对双腿接触平台的奖励。导致智能体在所有评估回合中快速失败（early terminal），无法学到稳定着陆。本修改新增 `landing_reward` 组件，利用双腿接触标志结合距离、速度构建连续因子，对接近成功着陆的状态给予正向奖励。这填补了“成功信号”空白，引导智能体在接近平台时尝试轻柔着陆，而非硬撞击。

- **数学形式**：`landing_reward = w * min(nl_contact, nr_contact) * max(0, 1 - dist/thr_dist) * max(0, 1 - speed/thr_speed)`，为有界连续信号。
- **系数校准**：权重 `w = 0.05`，使得在完全着陆状态（两腿接触、距离≈0、速度≈0）时每步获得 0.05，量级与 progress 的预期正收益相当，避免过分主导总奖励。距离阈值 0.3、速度阈值 0.5，确保只在靠近目标且低速时激活，稀疏但渐近。
- **组件职责**：之前 `landing_reward` 不存在，该组件从零开始学习，不干扰已有结构。

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

    # New landing progress reward
    contact_factor = min(nl_contact, nr_contact)  # requires both legs in contact
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
- **audit**: 遗漏成功着陆的显式奖励；现有信号对稳定落地引导不足。
- **behavior**: agent 快速终止（early terminal），所有回合 score<-50，未能学习着陆。
- **signal**: 缺少正向“着陆成功”梯度，仅靠 progress 和惩罚难以收敛至安全着陆。
- **level**: Level 2（添加新组件 `landing_reward`）
- **hypothesis**: 引入基于双腿接触、距离和速度的连续有界奖励，将提供更直接的着陆目标梯度，帮助策略向轻柔着陆方向优化，减少 hard crash。
- **risk**: 奖励值可能使 agent 过度追求接触而忽略姿态稳定，但权重保守且 factor 连续，预计能被后续迭代校正。
