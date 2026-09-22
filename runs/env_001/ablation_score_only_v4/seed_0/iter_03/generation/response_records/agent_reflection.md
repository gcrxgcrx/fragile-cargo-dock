# Response Record

# 设计理由
## 审计与诊断
当前 agent 的平均 episode 长度 992.65，仅有 1/20 成功触发终止（其余均被截断），说明它学会了接近目标、降低速度以获取进度奖励和 gate 增益，但并未真正完成降落到着陆垫的任务。代码缺少与“双脚接触”这一关键着陆前兆直接相关的正向信号——现有 `landing_gate` 仅根据距离、速度、角速度缩放运动奖励，没有将接触状态引入奖励积极引导。此外，缺少对动作/引擎使用的节省燃料激励，但当前最紧迫的问题是 **成功停靠率几乎为零**，因此本轮干预聚焦于**添加一个鼓励最终着陆接触的奖励组件**。

根据 Level 2 结构变换：“缺信号 → add 新组件”。我们将添加 `landing_reward`，在飞行器双脚接触地面且速度、角速度足够低时提供逐步奖励，激励它完成停靠。

## 设计校准
- 阈值设在终止边界附近：速度阈值 0.2（接近静止，略高于可能触发 settle 的细微值），角速度阈值 0.1。  
- 组件形式：乘积（不会塌缩，因为条件在预期情况可同时满足，且远距离时 `contact_avg=0` 使奖励为 0，不干扰探索）。  
- 系数 0.5：设想到达接触阶段时，每一步该奖励约 0.3~0.5，即使叠加到 `motion_reward` 上也远低于 2× 主信号；并且当 gate 拉满时主信号可能接近 0，此时 `landing_reward` 成为主要信号，引导停靠完成。  
- 总惩罚负担未变，安全。

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

    # --- 软着陆 gate ---
    dist_thresh = 0.3
    speed_proxy_thresh = 0.3
    ang_thresh = 0.2
    f_dist  = max(0.0, 1.0 - dist_new / dist_thresh)
    f_speed = max(0.0, 1.0 - speed_new / speed_proxy_thresh)
    f_ang   = max(0.0, 1.0 - abs(ang_vel) / ang_thresh)
    landing_gate = (f_dist + f_speed + f_ang) / 3.0

    # --- 新增：最终着陆接触奖励 ---
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0          # 双脚接触程度 [0,1]
    speed_slow_factor = max(0.0, 1.0 - speed_new / 0.2)      # 速度越小越接近 1
    ang_slow_factor   = max(0.0, 1.0 - abs(ang_vel) / 0.1)  # 角速度越小越接近 1
    landing_reward = contact_avg * speed_slow_factor * ang_slow_factor * 0.5

    # --- 组合 ---
    motion_reward = progress_reward + velocity_penalty + angular_penalty
    total_reward = landing_gate * motion_reward + landing_reward

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angular_penalty": angular_penalty,
        "landing_gate": landing_gate,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少对双脚接触的着陆引导信号，导致 agent 只学会靠近但不降落。  
- **behavior**: agent 在目标附近低速徘徊，持续累积小幅度进度奖励，几乎从不触发终止（停靠失败）。  
- **signal**: 需要接触感知的正向奖励，在双脚触地且几乎静止时给予明确强化。  
- **level**: Level 2（结构变换：添加新组件）。  
- **hypothesis**: 通过新增 `landing_reward`，agent 在靠近目标后将被引导去主动接触地面并维持静止，从而大幅提升成功终止率。  
- **risk**: 若系数过大，可能导致 agent 过早尝试接触（如高速撞击）以获取奖励；但经过速度、角速度两重门控，该风险可控。
