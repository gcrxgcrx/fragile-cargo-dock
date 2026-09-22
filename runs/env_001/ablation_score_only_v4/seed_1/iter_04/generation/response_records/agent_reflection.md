# Response Record

# 设计理由
当前奖励函数在迭代至第三轮时已接近收敛（score 202.18，超过目标 200），且趋势正确（score 从 11→14→202，len 从 780→982→468）。主要问题在于 **landing_success 乘积结构在精确接触发生前梯度坍塌**：当 `dist > 1.5` 时 `near_target = 0` 导致组件完全消失，迫使 agent 在无信号指导下从远距直接跳到近距着陆。根据公式切换指南的"乘积 proxy 经常塌缩为 0 → 几何平均"模式，需要将 `landing_success` 的裸乘积改为有连续梯度的形式。同时保留已证明有效的其余组件（goal_proximity、velocity_damping、angle 约束）。

**修改内容**: 单一组件 — `landing_success` 从 `2.0 * contact_score * near_target` 改为 `2.0 * (contact_score * near_target) ** 0.5`  
**数学形式**: 几何平均替代裸乘积，确保当 `contact_score` 高但 `near_target` 低时仍有非零梯度  
**系数校准**: 保持系数 2.0 不变，几何平均在 `contact_score=1, near_target=1` 时给出相同最大值 2.0，但在中间区域降低幅度以维持学习信号

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    x = next_obs[0]           # horizontal position (relative to target)
    y = next_obs[1]           # vertical position (relative to pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle = next_obs[4]       # body angle
    angvel = next_obs[5]      # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    speed = (vx**2 + vy**2)**0.5
    dist = (x**2 + y**2)**0.5

    # 1. Decomposed proximity
    vertical_proximity = 2.718281828 ** (-abs(y))          # [0,1]
    horizontal_alignment = 2.718281828 ** (- (x**2))       # [0,1]
    goal_proximity = vertical_proximity * horizontal_alignment

    # 2. Landing success – geometric mean for non-collapsing gradient
    contact_score = (left_contact + right_contact) / 2.0   # [0, 0.5, 1]
    near_target = max(0.0, 1.0 - dist / 1.5)               # [0,1], hinge at 1.5m
    landing_success = 2.0 * (contact_score * near_target) ** 0.5  # geometric mean prevents collapse to 0

    # 3. Velocity damping – reduced coefficient
    velocity_damping = -0.05 * speed * (2.718281828 ** (-dist))

    # 4. Orientation stability
    angle_penalty = -0.1 * (angle ** 2)
    angvel_penalty = -0.1 * (angvel ** 2)

    # Aggregate reward
    total_reward = goal_proximity + landing_success + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'landing_success': landing_success,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止条件 crash/horizontal_outside ≈ 高度异常与水平位置超出，当前代码无相关前兆软惩罚；成绩已达目标但 exploit margin 可进一步收窄。
- **behavior**: agent 在部分 episode 成功着陆，但 landing 过程仍有过渡依赖 sparse contact 信号的迹象（avg len 468 已较短，但 terminated 19/20 表明多数 episode 仍以失败终止）。
- **signal**: `landing_success` 裸乘积导致远距梯度消失，需要在保持相同最大值的同时提供连续引导。
- **level**: Level 2 — 结构变换（裸乘积 → 几何平均）
- **hypothesis**: 几何平均形态让 agent 在接近目标但尚未接触时仍能获得非零的 landing_success 信号，使得 `goal_proximity` 学习到的接近行为能被更顺畅地过渡到接触着陆，从而降低 crash/out-of-bounds 率。
- **risk**: 几何平均在极端情况（如 `contact_score=0`）下仍为零，但这与物理事实一致；近距区的信号略弱可能减慢最终收敛速度，但已用系数 2.0 保持最大激励。
