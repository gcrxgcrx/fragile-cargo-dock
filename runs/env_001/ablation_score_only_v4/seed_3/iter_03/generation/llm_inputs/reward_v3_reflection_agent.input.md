# 1. Search objective
- target_score: 200.000000
- current_score: -119.472383
- gap_to_target: 319.472383
- target_achievement_ratio: -59.736%

# 2. 上一轮奖励函数代码（该轮得分: -119.472383）
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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=-119.472383, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.892654, -95.474620]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
智能体需要控制一个 2D 飞行器从视口上方初始位置出发，最终稳定停靠在中央目标平台上。  
**主目标**：安全、准确地降落在目标平台上并稳定停靠（settled）。  
**次目标**：在满足主目标的前提下尽量减少燃料消耗、缩短到达时间。  
**非任务目标**：单纯追求快、单纯节省燃料而不考虑着陆安全，都不可取。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（通常连续空间默认）
- obs[0]: x_position，水平坐标（相对于目标平台中心），reward_usable: true
- obs[1]: y_position，垂直坐标（相对于平台高度），reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾斜角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作 0: no_engine，不点火
- 动作 1: left_orientation_engine，点燃一侧姿态发动机，产生偏航力矩
- 动作 2: main_engine，点燃主发动机，向下喷气提供向上推力（抵抗重力）
- 动作 3: right_orientation_engine，点燃另一侧姿态发动机，产生反向偏航力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - body_not_awake_or_settled：机体进入休眠或判定为已稳定停靠（可能结合接触和低运动状态），此条件终止可视为潜在成功。
- failure-like termination:
  - crash_or_body_contact：机体主体与地面碰撞或过于猛烈接触导致坠毁。
  - horizontal_position_outside_viewport：水平方向超出视口边界。
- ambiguous termination:
  - body_not_awake_or_settled 在没有足够上下文时也可能是失败（例如倒置卡死），但更常指向成功。
- truncation: 未提及，无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 任何假设的 "success"、"failure"、"landed"、"crash" 标志均不存在于 info 中

## 7. 可用于奖励函数的信号
- position: next_obs[0]（x 误差），next_obs[1]（y 误差）
- velocity: next_obs[2], next_obs[3]
- orientation: next_obs[4]（角度），next_obs[5]（角速度）
- contact: next_obs[6], next_obs[7]
- action/engine: action 取值 0/1/2/3（可推导是否开主发动机、姿态发动机）

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
| 1 | proximity_reward + safe_contact_bonus + soft_landing_penalty | 5.67 | 5.67 | 0.00 | 84.40 |
| 2 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -119.47 | 5.67 | -125.14 | 68.35 |