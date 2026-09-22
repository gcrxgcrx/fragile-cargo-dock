# 设计理由

本轮干预 `soft_landing_penalty`，将其从**无界二次惩罚**改为**有界饱和形式**，同时将门控阈值从 `1.0` 放宽至 `2.0`。

**为什么改它？**  
- 原 `dynamics_cost = vx² + vy² + angle² + ang_vel²` 在高速/高角速度时可达 50-100+，乘 `-0.5` 后单步惩罚可达 **-25 到 -50**。这淹没了所有其他信号（proximity ≈ -2.0~0, safe_contact_bonus max +2.0），导致 agent 的奖励被噪声主导，无法学到稳定着陆。  
- 同时，`landing_threshold=1.0` 使 penalty 仅在 `y_err < 1.0` 时激活——也就是说 agent 直到几乎触地时才突然受到巨量惩罚，没有梯度空间来减速。

**新数学形式：**  
每个动力学分量用 \( x^2 / (1 + x^2) \) 压缩到 `[0, 1)`，四项总和有界 `[0, 4)`，penalty 范围 `[-2, 0)`。  
门控阈值提升至 `2.0`，让 penalty 在 `y_err=2.0` 时从 0 开始线性渐入，agent 有充足时间在高空开始减速。

**系数校准：**  
- 主信号 `proximity_reward ≈ 0.5~2.0`（绝对值），penalty 通常 `-0.1~-0.5`，极端饱和 `-2.0`。  
- 惩罚负担 ≤ 主信号 30%，满足设计校准。

**未改动的已知缺口（留待后续迭代）：**  
- `proximity_reward` 全程为负，缺乏正向引导（后续可凸化）。  
- 缺少对 `horizontal_position_outside_viewport` 的显式边界前兆（后续可加 x_err hinge）。  
- `safe_contact_bonus` 仅着陆瞬间触发，active_rate 可能极低（暂观察本轮改动是否间接改善着陆行为）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation after action
    x_err = next_obs[0]   # horizontal position error relative to pad center
    y_err = next_obs[1]   # vertical position error (height above pad)
    vx = next_obs[2]      # horizontal velocity
    vy = next_obs[3]      # vertical velocity
    angle = next_obs[4]   # body tilt
    ang_vel = next_obs[5] # angular velocity
    left_contact = next_obs[6]  # left leg touch (0 or 1)
    right_contact = next_obs[7] # right leg touch (0 or 1)

    # ---------- Role 1: proximity_to_target (main learning signal) ----------
    dist = (x_err**2 + y_err**2)**0.5
    proximity_reward = -2.0 * (dist / (1.0 + dist))

    # ---------- Role 2: soft_landing_dynamics (stability constraint) ----------
    # Gate: activate gradually as vertical error decreases below threshold
    landing_threshold = 2.0
    gate = max(0.0, 1.0 - abs(y_err) / landing_threshold)
    # Bounded saturation function: x^2 / (1 + x^2) maps unbounded error to [0, 1)
    d_vx = vx**2 / (1.0 + vx**2)
    d_vy = vy**2 / (1.0 + vy**2)
    d_angle = angle**2 / (1.0 + angle**2)
    d_angvel = ang_vel**2 / (1.0 + ang_vel**2)
    dynamics_cost = d_vx + d_vy + d_angle + d_angvel  # bounded in [0, 4)
    soft_landing_penalty = -0.5 * gate * dynamics_cost

    # ---------- Role 3: safe_contact_encouragement (soft task completion proxy) ----------
    both_legs_contact = left_contact * right_contact  # 1 if both legs touch
    x_tol = 0.3
    y_tol = 0.3
    v_tol = 0.2
    angle_tol = 0.1
    angvel_tol = 0.1
    factor_x = max(0.0, 1.0 - abs(x_err) / x_tol)
    factor_y = max(0.0, 1.0 - abs(y_err) / y_tol)
    factor_vx = max(0.0, 1.0 - abs(vx) / v_tol)
    factor_vy = max(0.0, 1.0 - abs(vy) / v_tol)
    factor_angle = max(0.0, 1.0 - abs(angle) / angle_tol)
    factor_angvel = max(0.0, 1.0 - abs(ang_vel) / angvel_tol)
    mean_safety = (factor_x + factor_y + factor_vx + factor_vy + factor_angle + factor_angvel) / 6.0
    safe_contact_bonus = both_legs_contact * mean_safety * 2.0

    total_reward = proximity_reward + soft_landing_penalty + safe_contact_bonus

    components = {
        "proximity_reward": proximity_reward,
        "soft_landing_penalty": soft_landing_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 原 `soft_landing_penalty` 在低空高速时单步可达 -50，淹没主信号；gate 阈值太紧（1.0）无减速缓冲区；缺少视口边界前兆信号。
- **behavior**: agent 平均 84.4 步终止，score 在 [-50, 40] 宽幅波动，被 penalty 爆炸主导，未能学会稳定着陆。
- **signal**: `soft_landing_penalty` 过强且无界，主导 reward 方差；`proximity` 全程为负缺乏正向引导（暂未改）。
- **level**: Level 2 — 结构变换（无界二次惩罚 → 有界饱和形式 + 门控阈值放宽）
- **hypothesis**: 有界 penalty 消除极端负值对 value 的污染，放宽 gate 提供减速梯度空间，agent 将学到在接近地面时主动减速并稳定姿态，从而更频繁触发 safe_contact_bonus，score 上升。
- **risk**: penalty 上限降低后，agent 可能以较高速度着陆（硬着陆），若 crash 终止发生将限制 len 增长；需观察 terminated 原因是否从 penalty-explosion 转向 crash。