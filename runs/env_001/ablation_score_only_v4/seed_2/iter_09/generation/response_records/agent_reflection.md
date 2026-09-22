# Response Record

# 设计理由
**改动组件**：`orientation_penalty` 从全局二次惩罚改为局部门控形式，并降低系数。  
**数学形式**：原来 `-w_angle * abs_angle^2` 全程生效，导致探索阶段累积大量负分（估算每步约 -0.1，908 步总计约 -90，占主导）。现在将其替换为 `-0.15 * abs_angle^2 * max(0, 1 - next_dist)`，仅在距离目标平台 <1 的区域内激活，且系数由 0.5 降至 0.15，确保总惩罚负担 ≤ 主信号 per‑step 的 0.5x。  
**为什么有效**：在远离平台的绝大多数时间，agent 可以自由调整姿态进行探索，不会因倾斜被持续惩罚；当接近着陆区时，通过连续的 gate 引导竖直，避免撞击。这种“全局转局部”的结构变换缓解了惩罚支配问题，让 progress、landing 等其他信号有机会主导学习。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- unpack observations --------------------
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # -------------------- distance to target pad --------------------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # -------------------- derived quantities --------------------
    speed = (nvx**2 + nvy**2) ** 0.5
    abs_angle = abs(n_angle)

    # -------------------- weights --------------------
    w_progress = 2.0
    w_vel = 0.2
    w_angvel = 0.1
    w_angle = 0.15          # reduced and now gated
    w_landing = 2.0
    k_speed = 2.0
    k_angle = 2.0

    # -------------------- 1. progress reward --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (local gate) --------------------
    local_gate = max(0.0, 1.0 - next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * local_gate
    ang_penalty = -w_angvel * (n_angvel**2) * local_gate

    # -------------------- 3. upright orientation incentive (gated) --------------------
    orientation_penalty = -w_angle * (abs_angle**2) * local_gate

    # -------------------- 4. safe landing reward --------------------
    contact_sum = n_left_contact + n_right_contact
    speed_bonus = max(0.0, 1.0 - k_speed * speed)
    angle_bonus = max(0.0, 1.0 - k_angle * abs_angle)
    landing_factor = 1.0 + speed_bonus + angle_bonus
    landing_reward = w_landing * contact_sum * landing_factor

    # -------------------- total reward --------------------
    total_reward = (progress_reward +
                    vel_penalty +
                    ang_penalty +
                    orientation_penalty +
                    landing_reward)

    components = {
        "progress": progress_reward,
        "velocity_penalty": (vel_penalty + ang_penalty),
        "orientation_penalty": orientation_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少燃料节省（action penalty）与水平边界软信号；orientation_penalty 全程生效且系数过大，主导负总分。
- **behavior**: agent 徘徊近千步，得分累计为负，说明探索被惩罚压制，未学会着陆。
- **signal**: 全时姿态惩罚过强，应该仅在目标附近进行约束。
- **level**: Level 2
- **hypothesis**: 将 orientation_penalty 改为局部门控可释放远距离探索自由度，同时用连续 gate 引导接近阶段竖直，预期提升成功着陆概率。
- **risk**: 如果 agent 在高空忽视姿态控制，到达近距离时角度可能过大难以快速纠正，导致 crash；但 gate 的线性增强应提供足够梯度。
