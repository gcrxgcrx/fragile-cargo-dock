# Response Record

# 设计理由
上一轮训练后 agent 全部被截断（len=1000），得分 -13.4，说明策略几乎不动或随机徘徊，未能着陆。审计发现 **velocity_penalty 是全局惩罚**（通过 1/(1+k*next_dist) 缩放仍无处不在），它抵消了 progress 给予的正梯度，使得 agent 为了避免惩罚而选择不移动。  
本轮修改 **velocity_penalty 组件**：  
- 将门控从全局缩放改为 **硬局部门控**：`gate = max(0, 1.0 - next_dist)`，当距离 > 1.0 时惩罚完全为零，agent 可以在远处自由加速接近目标。  
- 同时将系数 `w_vel`、`w_angvel` 降为原来约 1/10，确保靠近目标时减速信号温和，不压制着陆尝试。  
这是 Level 2 的结构变换（全局惩罚→局部门控），只改一个组件。

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
    w_vel = 0.2          # was 2.0, now much smaller & local
    w_angvel = 0.1       # was 0.5
    w_angle = 0.5
    w_landing = 1.0
    k_speed = 5.0
    k_angle = 5.0

    # -------------------- 1. progress reward (unchanged) --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (local gate) --------------------
    # gate: 1 when dist=0, 0 when dist>=1.0
    local_gate = max(0.0, 1.0 - next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * local_gate
    ang_penalty = -w_angvel * (n_angvel**2) * local_gate

    # -------------------- 3. upright orientation incentive (unchanged) --------------------
    orientation_penalty = -w_angle * (abs_angle**2)

    # -------------------- 4. safe landing reward (unchanged) --------------------
    contact_sum = n_left_contact + n_right_contact
    speed_factor = 1.0 / (1.0 + k_speed * speed)
    angle_factor = 1.0 / (1.0 + k_angle * abs_angle)
    landing_reward = w_landing * contact_sum * speed_factor * angle_factor

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
- **audit**: 终止前兆（低速、直立、双支撑腿接触）已部分覆盖，但缺少越过水平边界的预警；全局速度惩罚抑制了前期移动，属于过度约束。
- **behavior**: agent 几乎不移动或极小幅度徘徊，len=1000 全截断，未能着陆。
- **signal**: velocity_penalty 在远离目标时仍有非零值，导致净 reward 偏向“不动”，progress 信号被淹没。
- **level**: Level 2
- **hypothesis**: 将速度惩罚改为严格局部门控后，agent 在远处不再受罚，可以自由加速接近目标；靠近时减速要求仍保留，从而恢复向目标移动的动机，提高得分和着陆率。
- **risk**: 若 agent 在远处高速移动导致角度失控或越界，可能引起 crash，但这种 crash 可由环境直接终止，反而提供了自然边界，后续可加横向位置预警来缓解。
