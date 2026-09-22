# 1. Search objective
- target_score: 200.000000
- current_score: 11.006412
- gap_to_target: 188.993588
- target_achievement_ratio: 5.503%

# 2. 上一轮奖励函数代码（该轮得分: 11.006412）
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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=11.006412, len=780.500000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-79.395903, 153.616317]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务：控制一个2D刚体（搭载推力器）从视口顶部中央附近出发，**到达并稳定、安全地停靠在画面中央的目标垫上**。  
次任务：在满足主任务的前提下，**尽可能少地使用引擎推力**（省燃料），并**尽快完成**。  
不应混淆的目标：纯粹的燃料最小化或最短时间不应牺牲安全接触与姿态稳定，着陆必须平稳。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 （推断，标准连续空间）
- obs[0]: `x_position`，水平相对坐标（可能是到目标垫中心的横向距离），可用于奖励（希望→0）
- obs[1]: `y_position`，垂直相对坐标（相对于垫的高度），可用于奖励（希望→0）
- obs[2]: `x_velocity`，水平线速度，可用于奖励（希望→0）
- obs[3]: `y_velocity`，竖直线速度，可用于奖励（着陆时希望→0或很小负值，但通常希望为0）
- obs[4]: `body_angle`，刚体朝向角度，可用于奖励（希望→0，保持直立）
- obs[5]: `angular_velocity`，角速度，可用于奖励（希望→0）
- obs[6]: `left_support_contact`，左侧支撑接触标志，0或1，可用于奖励（着陆成功时期望=1）
- obs[7]: `right_support_contact`，右侧支撑接触标志，0或1，可用于奖励（着陆成功时期望=1）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`（无推力，依靠惯性滑行）
- action 1: `left_orientation_engine`（向左定向推力，可能影响角速度或横向平移）
- action 2: `main_engine`（主推力，通常向上喷气，产生主要向上加速度，对抗重力）
- action 3: `right_orientation_engine`（向右定向推力，与左引擎对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled`  
  含义：刚体不再“醒着”（动能很低、角度稳定、接触垫子后进入休眠或标记为已停靠）。推测当成功着陆并稳定后触发。
- **failure-like termination**:
  - `crash_or_body_contact`：身体发生异常碰撞（可能并非目标垫，例如撞到地面、墙壁或其他部分）
  - `horizontal_position_outside_viewport`：水平位置超出允许范围，飞出视野
- **ambiguous termination**: 无
- **truncation**: 未给出 max steps 信息，可能不存在时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 为空，无 `success` 字段）
- explicit_failure_flag_available: **false** （无 `failure` 字段）
- allowed_info_fields: **无** （`info = {}`，禁止使用任何额外信息）
- forbidden_or_uncertain_info_fields: 任何未在 source 中声明为可用的字段，包括 `success`, `failure`, `terminal_observation` 等

## 7. 可用于奖励函数的信号
- **position**: `obs[0]` (x), `obs[1]` (y) — 可计算到目标垫的距离，鼓励趋近于0
- **velocity**: `obs[2]` (vx), `obs[3]` (vy) — 可惩罚绝对值以减慢速度，特别是接近目标时
- **orientation**: `obs[4]` (angle) — 可惩罚偏离直立的角度
- **angular velocity**: `obs[5]` — 可惩罚快速旋转
- **contact**: `obs[6]` (left), `obs[7]` (right) — 用于设计着陆成功奖励或接触条件，期望双侧均为1且速度很小
- **action / engine usage**: `action` 本身 — 可惩罚使用主引擎或所有引擎的情形，促进燃料效率
- **other**: 可从位置、速度、接触组合出复合信号（如“已着陆标志”：两个接触且速度/角度在阈值内）

# 6. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Score-Only Reward Memory

| iter | skeleton | score | best | delta | len |
|---:|---|---:|---:|---:|---:|
| 1 | angle_penalty + angvel_penalty + goal_proximity + safe_contact_proxy + velocity_damping | 11.01 | 11.01 | 0.00 | 780.50 |