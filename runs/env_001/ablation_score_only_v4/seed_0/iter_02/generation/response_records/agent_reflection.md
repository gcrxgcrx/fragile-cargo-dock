# Response Record

# 设计理由
**干预层级：Level 2 — 结构变换**

**问题诊断：**
当前 agent 在 1000 步全被截断（truncated=20/20），未出现提前终止。得分 -113，说明飞行器在徘徊但未能完成着陆。`landing_proxy` 组件（active rate 约 60-80%）作为加性项混入 total_reward，但其信号被 `progress_reward` 的噪声淹没。靠近目标时，`landing_proxy` 仅贡献约 0.3 的奖励，而 `progress_reward` 每步波动在 ±0.1 量级——着陆信号太弱，无法驱动精准停靠行为。

**修改：将 `landing_proxy` 从加性组件改为乘性 gate**
- 把 `landing_proxy` 变成 gate factor 乘到 `progress_reward + velocity_penalty + angular_penalty` 上。
- 当飞行器远离目标或高速/高角速度时，gate 接近 0，抑制所有奖励（此时 agent 应专注于靠近目标）。
- 当飞行器进入"着陆区"（dist<0.3, speed<0.3, ang<0.2），gate 接近 1，此时 `progress_reward` 仍有微小正向梯度，引导最后精调。
- 保留 `landing_proxy` 本身作为独立组件用于监控，但不加性计入 reward。

**系数校准：**
- `progress_reward` 保持不变（1.0 * delta_dist）。
- gate 计算保持原有 `landing_proxy` 公式（三个 bounded factor 平均）。在 safe 但未着陆区域（dist≈0.5, speed≈0.4, ang≈0.3），gate≈0.3，符合"不理想但安全时 gate≥0.3"的设计校准要求。
- 惩罚负担：`velocity_penalty` per-step ≈ -0.01（speed≈1.0 时超速 penalty 约 -0.12），`angular_penalty` per-step ≈ -0.003。主信号 per-step ≈ 0.1。总惩罚负担 < 0.15x 主信号，符合 ≤0.5x 要求。

**与累积记录的关系：**
这是第一轮反思，无历史失败模式可参考。当前设计的问题在于 `landing_proxy` 作为加性项时信号量级不足。改为乘性 gate 后，gate 从"奖金"变成"解锁器"，强化了 land-or-nothing 的语义。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 计算到目标的距离 ---
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_dist = dist_old - dist_new                      # 正值表示靠近目标

    progress_reward = 1.0 * delta_dist                    # 主学习信号

    # --- 速度软约束（超过阈值才惩罚）---
    speed_new = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_threshold = 0.5
    vel_excess = max(0.0, speed_new - speed_threshold)
    velocity_penalty = -0.5 * (vel_excess ** 2)           # 二次惩罚，仅作用于超速部分

    # --- 角速度轻量惩罚 ---
    ang_vel = next_obs[5]
    angular_penalty = -0.1 * (ang_vel ** 2)               # 抑制剧烈旋转

    # --- 软着陆 gate：靠近 + 低速 + 姿态稳定 ---
    dist_thresh = 0.3
    speed_proxy_thresh = 0.3
    ang_thresh = 0.2
    f_dist  = max(0.0, 1.0 - dist_new / dist_thresh)
    f_speed = max(0.0, 1.0 - speed_new / speed_proxy_thresh)
    f_ang   = max(0.0, 1.0 - abs(ang_vel) / ang_thresh)
    landing_gate = (f_dist + f_speed + f_ang) / 3.0       # 平均避免乘积塌缩，gate ∈ [0,1]

    # --- 组合：gate 乘法到全部运动相关奖励 ---
    motion_reward = progress_reward + velocity_penalty + angular_penalty
    total_reward = landing_gate * motion_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angular_penalty": angular_penalty,
        "landing_gate": landing_gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少效率信号（动作空间=4，维度不高，可暂缓）；`landing_proxy` 作为加性组件信号量级不足（per-step≈0.3 vs 主信号波动±0.1），应改为乘性 gate。
- **behavior**: agent 在 1000 步内慢速徘徊，未能触发终止，徘徊阶段 `landing_proxy` 被 `progress_reward` 的噪声淹没。
- **signal**: 着陆信号太弱。加性 `landing_proxy` 无法在全局奖励中占据主导，导致 agent 只学靠近但不学停下。
- **level**: Level 2
- **hypothesis**: gate 乘到全部运动奖励后，远离目标时 reward≈0（gate≈0），靠近目标时 reward 逐渐解锁，agent 被迫学"必须同时做到靠近+减速+稳定"才能获得正奖励，从而提高早停率。
- **risk**: gate 在训练初期可能长时间为 0，导致奖励稀疏、探索变慢。缓解：`f_dist` 在 dist≈0.6 时已激活到 0.3，gate≈0.3 提供基础梯度。若下一轮 len 仍 1000 且 score 不变，则需进一步降低 speed_threshold 以扩大 gate 覆盖范围。
