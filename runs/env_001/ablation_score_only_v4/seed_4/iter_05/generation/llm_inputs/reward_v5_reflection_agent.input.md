# 1. Search objective
- target_score: 200.000000
- current_score: 261.390117
- gap_to_target: -61.390117
- target_achievement_ratio: 130.695%

# 2. 上一轮奖励函数代码（该轮得分: 261.390117）
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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=261.390117, len=325.700000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[148.015881, 300.230362]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个带推进器的 2D 刚体（初始在顶部附近并带有随机初始力）到达并稳定停靠在中央目标着陆区域（target pad）。稳定停靠要求接近水平位置、极低速、竖直姿态并保持双支撑接触。

次要目标：在保证安全停靠的前提下，尽可能快地完成停靠，同时尽可能减少引擎推力使用（节省燃料）。

不应混淆的目标：并非单纯的位置追踪或连续前进，着陆后的“存活/平衡”是结果而非持续目标，核心仍是到达并安全停靠。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position（相对目标 pad 的水平坐标），reward_usable: true
- obs[1]: y_position（相对 pad 高度的垂直坐标），reward_usable: true
- obs[2]: x_velocity（水平线速度），reward_usable: true
- obs[3]: y_velocity（垂直线速度），reward_usable: true
- obs[4]: body_angle（机体角度，0 表示竖直向上），reward_usable: true
- obs[5]: angular_velocity（角速度），reward_usable: true
- obs[6]: left_support_contact（左支撑触点状态，1.0=接触，0.0=未接触），reward_usable: true
- obs[7]: right_support_contact（右支撑触点状态，1.0=接触，0.0=未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（无推力）
- action 1: left_orientation_engine（激活左侧姿态控制引擎，产生旋转力矩，可能带微小线加速度）
- action 2: main_engine（激活主引擎，产生向上的推力，同时有不大的线加速度）
- action 3: right_orientation_engine（激活右侧姿态控制引擎，产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体静止并被认为已停靠，其同时要求接触地面且极低动能，隐含成功着陆）
- failure-like termination: crash_or_body_contact（剧烈碰撞或错误接触，如高速撞击地面或翻倒）、horizontal_position_outside_viewport（水平飞出边界）
- ambiguous termination: body_not_awake_or_settled 本身是成功信号，但 crash_or_body_contact 中可能包含“部分着陆但未稳定”的情形，但仍被归为失败终止
- truncation: 未提供明确的步数限制，但存在默认 horizon（例如 1000 步），在实际环境中可视为非失败截断，但本说明未提供该信息，按通常情况作为潜在截断处理（无 info 字段）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（无 info 字段，也没有显式 success 标志）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 在 step 中返回空字典 {}）
- forbidden_or_uncertain_info_fields: 一切 info 字段均不允许使用，因为源中未定义

## 7. 可用于奖励函数的信号
- position: x_position, y_position（相对目标 pad 的水平与垂直位移）
- velocity: x_velocity, y_velocity
- orientation: body_angle, angular_velocity
- contact: left_support_contact, right_support_contact（布尔化 1.0/0.0）
- action/engine: action 值 0/1/2/3 可用于判断是否使用推力
- other: 可基于上述信号合成距离、速度幅值、角度绝对值、是否着陆成功等衍生量

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
| 1 | fuel_penalty + goal_proximity + safe_landing_penalty | -11.79 | -11.79 | 0.00 | 1000.00 |
| 2 | fuel_penalty + landing_proxy + safe_landing_penalty | -37.08 | -11.79 | -25.28 | 1000.00 |
| 3 | landing_proxy + safe_landing_penalty | -9.21 | -9.21 | 0.00 | 1000.00 |
| 4 | potential_gain + safe_landing_penalty | 261.39 | 261.39 | 0.00 | 325.70 |