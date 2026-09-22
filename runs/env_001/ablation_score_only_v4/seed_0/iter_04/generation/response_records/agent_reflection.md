# Response Record

# 设计理由

**审计发现**：Landing_gate 当前定义为（f_dist + f_speed + f_ang）/ 3，其值在接近目标且低速时 **升高**，作为 motion_reward 的乘性因子。这会在最需要减速的着陆前阶段 **放大** progress 奖励，鼓励 agent 在低距离低速度时继续加速，违背“尽快稳定停靠”的原始目标。应将其逆转为**制动门控**，使条件越好（越近、越慢）则 motion_reward 被压制，由 landing_reward 主导最终停顿。

**修改内容**：将 `landing_gate` 改为 `braking_gate = 1.0 - (f_dist + f_speed + f_ang)/3.0`，其他组件保持不变。制动因子处于 [0,1]：远处 ≈ 0（不抑制 progress），近处且低速 ≈ 1（强烈抑制 progress），迫使 agent 在着陆区依赖 velocity_penalty 和 landing_reward 平稳停靠。

**系数校准**：无需更改任何系数，per-step 惩罚比重仍远低于主信号，hinge 阈值维持原值。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 计算到目标的距离 ---
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_dist = dist_old - dist_new

    progress_reward = 1.0 * delta_dist

    # --- 速度软约束 ---
    speed_new = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_threshold = 0.5
    vel_excess = max(0.0, speed_new - speed_threshold)
    velocity_penalty = -0.5 * (vel_excess ** 2)

    # --- 角速度轻量惩罚 ---
    ang_vel = next_obs[5]
    angular_penalty = -0.1 * (ang_vel ** 2)

    # --- 制动门控（替代原 landing_gate）---
    dist_thresh = 0.3
    speed_proxy_thresh = 0.3
    ang_thresh = 0.2
    f_dist  = max(0.0, 1.0 - dist_new / dist_thresh)
    f_speed = max(0.0, 1.0 - speed_new / speed_proxy_thresh)
    f_ang   = max(0.0, 0 - abs(ang_vel) / ang_thresh)   # 注意：角速度小则 f_ang 接近 1
    # 将原始 gate 逆转，变为制动需求因子
    braking_gate = 1.0 - (f_dist + f_speed + f_ang) / 3.0

    # --- 最终着陆接触奖励 ---
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    speed_slow_factor = max(0.0, 1.0 - speed_new / 0.2)
    ang_slow_factor   = max(0.0, 1.0 - abs(ang_vel) / 0.1)
    landing_reward = contact_avg * speed_slow_factor * ang_slow_factor * 0.5

    # --- 组合 ---
    motion_reward = progress_reward + velocity_penalty + angular_penalty
    total_reward = braking_gate * motion_reward + landing_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angular_penalty": angular_penalty,
        "braking_gate": braking_gate,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 原 landing_gate 在需要减速时放大 motion_reward，方向错误
- **behavior**: agent 成功着陆但可能保留过高终端速度
- **signal**: 缺乏着陆前抑制前进的制动信号
- **level**: Level 2
- **hypothesis**: 逆转门控方向可使 agent 在接近目标时自动减速，提升着陆稳定性
- **risk**: 若制动过早过强，可能延长任务时间，但现有速度惩罚会维持合理行动速度
