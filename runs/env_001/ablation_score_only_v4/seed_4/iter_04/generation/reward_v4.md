# 设计理由

第一轮反思的场景：当前奖励函数 `landing_proxy + safe_landing_penalty` 给出最终得分 -9.21，所有 episode 均被截断在 1000 步，既无成功着陆也无 crash 终止。行为表现是**agent 在任务空间中徘徊，有接触但未达成稳定着陆条件**。  
根因在于 `landing_proxy` 采用**绝对值奖励**：agent 只要处于“不太差”的状态就能获取正反馈，缺乏持续改善的紧迫梯度，尤其在接近目标时梯度消失，导致它满足于部分接触 (contact_bonus 独立加法) 而忽略了姿态、速度等维度的彻底优化。

本次修改选择 **Level 2 结构变换**：将 `landing_proxy` 从基于当前状态的绝对值奖励改为基于 **potential-shaped 差分** 的形式。新组件 `potential_gain` 计算前后两帧的 potential 差值，让 agent 每一步都能因“变得更好”而获得正反馈，同时将 contact 信号**乘入 potential** (避免加法破坏多目标平衡)。这种差分 shaping 不改变最优策略，但能大幅缓解“高原徘徊”问题。

数学形式：
```
potential(s) = pos_factor * speed_factor * angle_factor * contact_factor
  pos_factor    = 1 / (1 + √(x² + y²))
  speed_factor  = 1 / (1 + √(x_vel² + y_vel² + 0.1*ang_vel²))
  angle_factor  = max(0, 1 - |body_angle|/π)
  contact_factor= (left_contact + right_contact) / 2
reward = 2.0 * (potential(next_obs) - potential(obs))
```
scale=2.0 使一次完全改善 (potential 从 0 到 1) 产生 +2.0 奖励，足以盖过 safe_landing_penalty 的轻微惩罚，推动 agent 快速向完美姿态收敛。  
保留 `safe_landing_penalty` 未作改动，以符合“每轮只改一个组件”的约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x_cur, y_cur = obs[0], obs[1]
    xv_cur, yv_cur = obs[2], obs[3]
    angle_cur = obs[4]
    angv_cur = obs[5]
    lc_cur, rc_cur = obs[6], obs[7]

    # Unpack next observation
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    xv_nxt, yv_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angv_nxt = next_obs[5]
    lc_nxt, rc_nxt = next_obs[6], next_obs[7]

    # ---- Potential function ----
    def potential(x, y, xv, yv, angle, angv, lc, rc):
        dist = (x**2 + y**2) ** 0.5
        pos_factor = 1.0 / (1.0 + dist)
        speed = (xv**2 + yv**2 + 0.1 * angv**2) ** 0.5
        speed_factor = 1.0 / (1.0 + speed)
        angle_factor = max(0.0, 1.0 - abs(angle) / 3.14159265)
        contact_factor = (lc + rc) / 2.0   # [0,1]
        return pos_factor * speed_factor * angle_factor * contact_factor

    pot_curr = potential(x_cur, y_cur, xv_cur, yv_cur, angle_cur, angv_cur, lc_cur, rc_cur)
    pot_next = potential(x_nxt, y_nxt, xv_nxt, yv_nxt, angle_nxt, angv_nxt, lc_nxt, rc_nxt)

    potential_gain = 2.0 * (pot_next - pot_curr)

    # ---- Component B: Safe landing penalty (unchanged) ----
    y_pos = next_obs[1]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]

    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.01 * landing_gate * motion_cost

    # ---- Total reward ----
    total_reward = potential_gain + safe_landing_penalty

    components = {
        "potential_gain": potential_gain,
        "safe_landing_penalty": safe_landing_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前缺少持续改善梯度与失败前兆硬约束；绝对值奖励造成策略满足于局部奖励（接触加分）而不推动全局收敛。
- **behavior**: agent 未 crash 也未成功，徘徊 1000 步被截断；可能在中等高度或部分接触状态维持，缺乏完成着陆的紧迫感。
- **signal**: 缺“进步信号”（改善量），而原有绝对值信号在接近目标时梯度消失，且 contact 加法奖励弱化了姿态/速度维度的优化。
- **level**: Level 2
- **hypothesis**: 差分 reward 会给每一步的改善提供明确正反馈，推动 agent 从初始状态快速过渡到最佳着陆姿态；contact 乘入 potential 确保无接触时潜力为 0，引导下降。
- **risk**: 若环境存在“先降接触再调整”的困难，差分会放大接触瞬间的奖励跳跃，可能鼓励硬着陆；safe_landing_penalty 虽保留但系数小，需在下一轮根据 crash 情况校准。