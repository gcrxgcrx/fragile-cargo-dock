# Response Record

# 设计理由
本轮只修改 **contact_reward** 组件，将其从简单的二元接触奖励改为 **联合条件连续奖励**（safe landing incentive）。  
当前 agent 最终策略 `len≈874`，`terminated=5/20`，说明多数 episode 因截断结束而非自然终止（成功/失败）。可能原因是：agent 靠近目标后会输出支撑腿接触，但未能将速度和姿态减小到足够触发“body_not_awake_or_settled”终止，因而长时间在平台上移动而不 settle。  
原有的 `contact_reward` 只奖励接触本身，不奖励低速、小角度，导致接触后没有进一步减速动力。  
将组件改为：`landing_reward = w_landing * (left_contact + right_contact) * speed_factor * angle_factor`，其中  
`speed_factor = 1/(1+k_speed * speed)`，`angle_factor = 1/(1+k_angle * |angle|)`。这样在接触状态下，速度越低、姿态越竖直，奖励越高，形成连续梯度引导 agent 完成最终稳定着陆。

系数设置：`w_landing=1.0`，`k_speed=5.0`，`k_angle=5.0`，  
- 当速度≈0.2（较快）时 factor≈0.5；  
- 当角度≈0.2 rad (~11.5°) 时 factor≈0.5。  
最大可能奖励（零速零角且双接触）为 `2.0`，仅略高于原最大 `1.0`，但在非理想状态下奖励会衰减，鼓励改善。  
符合“单组件≤2x主信号”、“总惩罚负担合理”等设计校准。

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
    w_vel = 2.0
    w_angvel = 0.5
    w_angle = 0.5
    w_landing = 1.0
    k_proximity = 3.0
    k_speed = 5.0
    k_angle = 5.0

    # -------------------- 1. progress reward (improvement_delta) --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (soft, proximity-gated) --------------------
    proximity_gate = 1.0 / (1.0 + k_proximity * next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * proximity_gate
    ang_penalty = -w_angvel * (n_angvel**2) * proximity_gate

    # -------------------- 3. upright orientation incentive --------------------
    orientation_penalty = -w_angle * (abs_angle**2)

    # -------------------- 4. safe landing reward (joint condition proxy) --------------------
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
- **audit**: 缺失成功着陆的联合条件前兆（接触 + 低速 + 小角度），导致 agent 在接触后没有足够梯度减速并触发 `settled` 终止，多数 episode 以截断结束。
- **behavior**: agent 在接近目标时建立支撑腿接触，但保持较高速度/角度，未能进入稳定状态，因此长时间飞行（len≈874），仅 5/20 自然终止。
- **signal**: 缺少低速、小角度的着陆正向激励，原有 contact_reward 只奖励接触，不奖励减速。
- **level**: Level 2
- **hypothesis**: 将接触奖励改为速度‑角度联合门控的连续奖励，能提供清晰梯度迫使 agent 在接触后主动降低速度和倾斜角，从而提升 settled 终止率、缩短 episode 长度、并提高总体得分。
- **risk**: 若减速过快可能导致下降过于谨慎或悬停，但 progress_reward 仍会推动向前移动，总体可控。
