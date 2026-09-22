# 1. Search objective
- target_score: 200.000000
- current_score: -122.786648
- gap_to_target: 322.786648
- target_achievement_ratio: -61.393%

# 2. 上一轮奖励函数代码（该轮得分: -122.786648）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, l_contact, r_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, n_angle, n_ang_vel, nl_contact, nr_contact = next_obs

    current_dist = (x_pos**2 + y_pos**2) ** 0.5
    next_dist    = (nx_pos**2 + ny_pos**2) ** 0.5

    current_speed = (x_vel**2 + y_vel**2) ** 0.5
    next_speed    = (nx_vel**2 + ny_vel**2) ** 0.5

    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current

    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    if action == 0:
        fuel_cost = 0.0
    elif action == 2:
        fuel_cost = -0.1
    else:
        fuel_cost = -0.05

    crash_risk = max(0.0, -ny_vel - 0.1)
    proximity = max(0.0, 1.0 - next_dist / 0.3)
    crash_prevention = -0.05 * crash_risk * proximity

    # New landing progress reward
    contact_factor = min(nl_contact, nr_contact)  # requires both legs in contact
    dist_factor = max(0.0, 1.0 - next_dist / 0.3)
    speed_factor = max(0.0, 1.0 - next_speed / 0.5)
    landing_reward = 0.05 * contact_factor * dist_factor * speed_factor

    total = progress + orientation_penalty + fuel_cost + crash_prevention + landing_reward

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
        "crash_prevention": crash_prevention,
        "landing_reward": landing_reward
    }
    return float(total), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=-122.786648, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.466519, -100.568487]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个二维着陆问题：智能体控制一辆带主发动机和两个方向发动机的小车，从画面顶部中央附近开始（含随机初始冲量），尽快飞抵并平稳停靠在中央目标平台上。主要目标是到达目标并稳定着地；次要目标是尽量节省发动机推力（减少燃料消耗）并尽快完成。不应将姿态稳定或速度控制本身当作独立目标，它们是为安全着陆服务的。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32  
- 各维度含义（相对坐标系，原点为目标平台中心/高度）：
  - obs[0]: x_position – 相对于目标的水平坐标，reward_usable: true
  - obs[1]: y_position – 相对于平台高度的垂直坐标，reward_usable: true
  - obs[2]: x_velocity – 水平线速度，reward_usable: true
  - obs[3]: y_velocity – 垂直线速度，reward_usable: true
  - obs[4]: body_angle – 机体倾斜角，reward_usable: true
  - obs[5]: angular_velocity – 角速度，reward_usable: true
  - obs[6]: left_support_contact – 左支撑腿接触标志（0/1），reward_usable: true
  - obs[7]: right_support_contact – 右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0: no_engine – 不启动任何发动机
  - 1: left_orientation_engine – 启动左转向/姿态发动机
  - 2: main_engine – 启动主发动机
  - 3: right_orientation_engine – 启动右转向/姿态发动机

（动作是离散开关，无连续推力大小。每次 step 可选择执行一种发动机或待机）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` – 当身体进入休眠/稳定状态（可能表示已停稳在平台上）时终止，这最可能对应成功着陆。
- failure-like termination: `crash_or_body_contact` – 发生碰撞或身体其它部位不当接触（非支撑腿触地）时终止，代表坠毁；`horizontal_position_outside_viewport` – 飞出水平边界，失败。
- ambiguous termination: 无明确附加字段说明。
- truncation: 未提及 episode 截断，但在 RL 训练中超出最大步数会截断，此处不考虑。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无 success 键）
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，没有显式标志）
- forbidden_or_uncertain_info_fields: info 任何字段均不可用，不能去尝试猜测成功/失败标记。

注：终止原因本身在 compute_reward 函数外不可直接获知，只能通过 next_obs 的状态（如位置、速度、接触等）间接推断是否可能处于成功着陆状态。

## 7. 可用于奖励函数的信号
从 obs / next_obs / action 中可直接或间接使用的信号：
- 位置信号：
  - x, y（相对目标），可计算距离 `√(x²+y²)` 或分别处理
- 速度信号：
  - x_velocity, y_velocity，可计算合速度大小，尤其是垂直接近速度
- 姿态信号：
  - body_angle, angular_velocity，可用于维持竖直（angle≈0）
- 接触信号：
  - left_support_contact, right_support_contact（0/1），表示支撑腿是否着地，可作为着陆状态判断
- 动作信号：
  - 动作类型（0,1,2,3）可进行惩罚，因为每个非零动作代表一次发动机使用，消耗燃料。尤其可重点惩罚主发动机(2)，方向发动机(1,3)可较轻惩罚。
- 变化量信号：
  - 可由 obs→next_obs 计算速度变化、姿态变化来评估控制效果

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
| 1 | fuel_cost + orientation_penalty + progress | -121.77 | -121.77 | 0.00 | 68.30 |
| 2 | crash_prevention + fuel_cost + orientation_penalty + progress | -118.52 | -118.52 | 0.00 | 68.45 |
| 3 | crash_prevention + fuel_cost + landing_reward + orientation_penalty + progress | -122.79 | -118.52 | -4.27 | 68.30 |