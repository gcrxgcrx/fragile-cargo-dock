# 设计理由

## 第 0 步审计

### a) 终止 → 前兆
- **success-like `body_not_awake_or_settled`**：当刚体动能极低、接触垫子并稳定时触发。前兆信号应有：接近 `(x,y)≈0`、低速度、双侧接触。当前代码的 `safe_contact_proxy` (接触×低速度) 部分覆盖，但 `goal_proximity` 对 x 和 y 的无差别指数衰减无法区分"正在下落逼近"vs"正在远离"。
- **failure-like `crash_or_body_contact`**：身体撞到地面或墙壁。前兆信号应为：过大的 x 偏离、y 过负(低于垫子)或高速接触。当前代码无对应前兆信号。
- **failure-like `horizontal_position_outside_viewport`**：水平位置越界。前兆信号：|x| 过大、vx 向外。当前代码无预警，仅在 velocity_damping 中有间接速度惩罚但无位置门控。

### b) 目标 → 进度
- **任务目标**：到达并稳定安全停靠在垫上。需要：接近→软着陆→双侧接触→姿态稳定。
- 当前代码对"接近"有指数奖励（goal_proximity），对"软着陆"有 safe_contact_proxy 代理，对姿态有角度惩罚。**缺失**："接近"与"下降"的区分——agent 可以悬停在高处(y大)但水平接近(x小)获得高 goal_proximity，这不符合实际目标。

### c) 效率信号
- 动作维度 4 ≥ 4，当前无动作惩罚。但这不是最大问题，因为 fuel efficiency 是次目标。

### d) 僵尸组件
- 从训练反馈看，所有组件 active_rate=100%，无僵尸组件。但 angle_penalty 的 episode_sum_mean 预计约 -0.29/step (780步总分11, angle_penalty贡献巨大)，可能过强。

### e) 一句话结论
**当前 reward 漏了"垂直下降梯度"和"越界前兆"，且角度惩罚过强压倒了接近信号的贡献。**

---

## 第 1 步行为诊断

### agent 在做什么？
- score=11, len=780，terminated=9/20(可能多数是成功终止但得分低，或少数失败终止)，大部分 truncation。**agent 可能在徘徊**：维持姿态但未有效下降着陆。速度高会被 velocity_damping 惩罚，角度偏离被惩罚，但目标接近信号太弱→agent 选择"保持静止避免惩罚"而非"冒险下降着陆"。

### 干预哪个目标？
- **让 agent 愿意下降**。需要增强下降方向的梯度，使垂直接近(y→0)获得比水平漂移更大的奖励。

### 这个方向值得继续吗？
- 第一轮，无历史失败记录。Level 2 的结构修改即可。

---

## 第 2 步干预层级选择: Level 2 — 结构变换

**当前 goal_proximity = exp(-sqrt(x²+y²))**：x和y对距离的贡献对称。但任务目标的顺序是：先水平接近，再垂直下降(同时减速)。对称的指数衰减无法提供"下降时奖励增长更快"的梯度。

**变换**：将 goal_proximity 从对称距离衰减改为**分解式接近奖励**：
- `vertical_proximity = exp(-|y|)`：对垂直逼近提供密集梯度
- `horizontal_alignment = exp(-x²)`：水平对准，更窄的 sigma 促进精确对准
- 两者相乘，使 agent 必须先水平对准才能获得垂直下降的奖励

同时**削弱 angle_penalty** 至可容忍水平(降系数至 0.1)，防止姿态完美主义阻碍下降尝试。

---

## 第 3 步设计校准

1. **新惩罚系数**：angle_penalty 降为 0.1。主信号(vertical_proximity * horizontal_alignment)per-step 约 1.0 * 0.5 ≈ 0.5 量级。angle_penalty per-step 约 0.1 * 0.3² ≈ 0.009，远小于 0.3x 主信号。
2. **hinge阈值**：无新hinge。
3. **gate不塌缩**：无乘积gate，已用几何平均的替代方案。
4. **单组件≤2x主信号**：已验证。
5. **总惩罚负担**：仅 angle_penalty，per-step≈0.009，远小于主信号的 0.5x。

---

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

    # Exponential helpers (no imports allowed)
    exp_neg_speed = 2.718281828 ** (-speed)

    # 1. Decomposed proximity – vertical descent reward * horizontal alignment
    vertical_proximity = 2.718281828 ** (-abs(y))          # [0,1], peaks when y→0
    horizontal_alignment = 2.718281828 ** (- (x**2))      # [0,1], peaks when x→0, narrow sigma
    goal_proximity = vertical_proximity * horizontal_alignment  # joint incentive

    # 2. Safe contact proxy – joint condition: both contacts and low speed
    contact_proxy = left_contact * right_contact * exp_neg_speed

    # 3. Velocity damping – distance‑gated speed penalty (kept for safety)
    velocity_damping = -0.2 * speed * (2.718281828 ** (- (x**2 + y**2)**0.5))

    # 4. Orientation stability – reduced penalties to avoid dominating reward
    angle_penalty = -0.1 * (angle ** 2)      # was -0.5, now -0.1
    angvel_penalty = -0.1 * (angvel ** 2)    # kept at -0.1

    # Aggregate reward
    total = goal_proximity + contact_proxy + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'safe_contact_proxy': contact_proxy,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total), components
```

---

# 诊断摘要
- **audit**: 对称距离指数衰减无法区分水平徘徊和垂直下降，缺少越界前兆，角度惩罚过强压制接近信号
- **behavior**: 保持姿态但不下落着陆，在安全区域徘徊避免惩罚
- **signal**: 垂直下降方向梯度不足，水平对准和垂直逼近被混为一谈
- **level**: Level 2
- **hypothesis**: 分解垂直和水平接近信号，提供更强的下降梯度；削弱角度惩罚释放探索自由度，agent 将更愿意尝试下降着陆
- **risk**: 角度惩罚过弱可能导致着陆时姿态不稳，若后续出现则需适当调回至0.15–0.2范围