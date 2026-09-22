# reward_v1.py

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
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5

    # Exponential helpers (no imports allowed)
    exp_neg_dist = 2.718281828 ** (-dist)
    exp_neg_speed = 2.718281828 ** (-speed)

    # 1. Goal proximity – main learning signal (bounded positive reward)
    goal_proximity = exp_neg_dist

    # 2. Safe contact soft proxy – joint condition: both contacts and low speed
    contact_proxy = left_contact * right_contact * exp_neg_speed

    # 3. Velocity damping – distance‑gated speed penalty
    velocity_damping = -0.2 * speed * exp_neg_dist

    # 4. Orientation stability – quadratic penalties on angle and angular velocity
    angle_penalty = -0.5 * (angle ** 2)
    angvel_penalty = -0.1 * (angvel ** 2)

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

# reward_v1 设计说明

**任务画像**
- task_family: `navigation_goal_reaching`
- dynamics_subtype: `goal_approach_and_soft_contact`
- 核心目标：控制2D刚体到达目标垫中心并实现软着陆（接触稳定、速度近零、姿态竖直）。

**选择的奖励职责与公式算子**
1. **goal_proximity** – 主学习信号  
   - 信号：`next_obs[0], next_obs[1]`（位置）  
   - 公式：`exp(-dist)`（bounded_signal 形式，密集正奖励，距离越近奖励越大，最大为1）  
   - 理由：直接驱动agent向目标移动，有界且平滑，避免线性负奖励的尺度问题。

2. **velocity_damping** – 速度阻尼（距离门控）  
   - 信号：`next_obs[2], next_obs[3]`（速度） + `dist`  
   - 公式：`-0.2 * speed * exp(-dist)`（dense_state_signal 的 hinge 变体，但用指数门控实现距离依赖）  
   - 理由：在远处允许高速（exp(-dist) 很小，惩罚轻），靠近目标时惩罚指数上升，迫使减速软着陆。避免全程速度惩罚抑制初始加速。

3. **orientation_stability** – 姿态稳定性  
   - 信号：`next_obs[4]`（角度）, `next_obs[5]`（角速度）  
   - 公式：`-0.5 * angle²` 和 `-0.1 * angvel²`（quadratic_penalty）  
   - 理由：连续惩罚偏离直立的状态，防止翻转或单侧触地，同时保留微小姿态调整空间。

4. **safe_contact** – 软着陆完成近似信号  
   - 信号：`next_obs[6], next_obs[7]`（接触标志）, `speed`  
   - 公式：`left_contact * right_contact * exp(-speed)`（joint_condition_proxy 的连续化版本）  
   - 理由：仅在双支撑同时触地且速度极低时给出显著正奖励，为最终停靠提供额外引导。乘积因子虽使用二值接触，但结合连续的速度指数衰减，整体仍提供梯度（速度越小奖励越高）。权重与主奖励相当，但不主导学习过程。

**排除的角色及原因**
- `fuel_efficiency`：v1阶段先学会安全着陆，省燃料作为次要目标留到后续迭代。
- `time_to_land`：无显式时间惩罚需求，v1不加。
- `sparse_success_bonus` / `terminal_bonus`：环境无显式成功/失败标记（info为空），无法可靠判断终止原因，不能使用。
- `curriculum_weighting`：v1不依赖训练进度，动态权重留到后期。

**为什么没有使用 terminal_success_reward / terminal_failure_penalty**
- `explicit_success_flag_available` 为 false，`explicit_failure_flag_available` 为 false，无法获取可靠的终止类型，用观测推断易产生错误奖励。

**后续迭代可引入**
- 燃料效率（惩罚引擎使用，特别是主引擎）
- 时间效率（轻度每一步负值或基于进度的时间奖励）
- 更精细的接触约束（仅当体态也十分理想时才允许接触奖励，或使用高斯乘性因子）

**训练后应观察的潜在失败模式**
1. **撞击着陆**：velocity_damping权重不足时，agent可能以高速撞向目标垫，虽可能触发接触但姿态不稳或弹飞。
2. **姿态振荡**：orientation惩罚若不够，agent可能来回摆动，浪费燃料且无法双触地。
3. **hovering**：如果接触奖励权重过大而主距离奖励不够，agent可能学会在目标附近悬停但不敢着陆，需调整contact_proxy仅当双接触时才激活的特性可缓解。
4. **水平漂移**：因缺少显式水平边界惩罚，早期策略可能因速度大飞出视口导致失败；需依赖goal_proximity将其拉回中心。若episode提前终止，距离信号仍能引导。