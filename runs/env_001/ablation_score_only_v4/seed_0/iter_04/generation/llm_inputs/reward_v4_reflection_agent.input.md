# 1. Search objective
- target_score: 200.000000
- current_score: 208.814713
- gap_to_target: -8.814713
- target_achievement_ratio: 104.407%

# 2. 上一轮奖励函数代码（该轮得分: 208.814713）
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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=208.814713, len=573.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[176.878242, 240.771686]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
该环境是一个 2D 飞行器轨迹优化任务。一个刚体飞行器从视口顶部中央附近开始，带有随机的初始作用力。核心任务是控制飞行器的方向引擎和主引擎，使其飞到视口中央的目标着陆垫上，并尽快、稳定地停靠在垫上。次要目标是完成该过程所用的时间尽可能短，同时使用的发动机推力尽可能少。智能体需要学会逐步接近目标，减速，保持稳定的姿态，并安全接触着陆垫。不应将快速完成或省燃料与原目标（精准停靠）混淆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 浮点数（连续值，接触标志为 0.0 或 1.0）
- 各维度含义及奖励可用性：
  - obs[0] (x_position): 飞行器相对于目标垫的水平坐标。reward_usable: true
  - obs[1] (y_position): 飞行器相对于垫基准高度的垂直坐标。reward_usable: true
  - obs[2] (x_velocity): 水平线速度。reward_usable: true
  - obs[3] (y_velocity): 竖直线速度。reward_usable: true
  - obs[4] (body_angle): 机身朝向角度。reward_usable: true（但不建议作为强独立目标）
  - obs[5] (angular_velocity): 角速度。reward_usable: true（可用于姿态稳定性）
  - obs[6] (left_support_contact): 左侧支撑点是否接触的布尔标志（1.0/0.0）。reward_usable: true
  - obs[7] (right_support_contact): 右侧支撑点是否接触的布尔标志（1.0/0.0）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作详细说明：
  - 动作 0 (no_engine): 不启动任何引擎，依靠惯性滑行。
  - 动作 1 (left_orientation_engine): 点燃一个姿态引擎，用于改变飞行器的方向/角度。
  - 动作 2 (main_engine): 点燃主推进引擎，沿机头方向施加推力，用于移动。
  - 动作 3 (right_orientation_engine): 点燃与动作1相对的姿态引擎，用于反向调整姿态。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 可能意味着飞行器已经静止且可能已停靠，但具体成功条件未知，没有明确的“成功”标签。
- failure-like termination:
  - `crash_or_body_contact` —— 碰撞或非目标位置接触（可能表示严重撞击或侧翻）。
  - `horizontal_position_outside_viewport` —— 水平位置超出视口，即飞出边界。
- ambiguous termination: `body_not_awake_or_settled` 可能为成功（停稳在垫上），也可能为失败（因故障卡住无人为动作），但目前无附加信息佐证。
- truncation: 无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 中无相关字段）
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，且不得假设存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) —— 直接给出相对目标垫的水平、垂直距离
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) —— 可用于阻尼或安全约束
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) —— 可用于姿态稳定性，但无目标角度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) —— 着陆接触标志
- action/engine: 当前动作（离散值 0–3），可用于惩罚推力使用
- other: 可组合以上信号构建复合奖励，例如距离、速度、接触的联合条件

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
| 1 | angular_penalty + landing_proxy + progress_reward + velocity_penalty | -113.14 | -113.14 | 0.00 | 1000.00 |
| 2 | angular_penalty + landing_gate + progress_reward + velocity_penalty | 82.57 | 82.57 | 0.00 | 992.65 |
| 3 | angular_penalty + landing_gate + landing_reward + progress_reward + velocity_penalty | 208.81 | 208.81 | 0.00 | 573.00 |