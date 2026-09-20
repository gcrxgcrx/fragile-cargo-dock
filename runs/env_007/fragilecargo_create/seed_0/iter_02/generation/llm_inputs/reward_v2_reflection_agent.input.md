# 1. Search objective
- target_score: 250.000000
- current_score: -2.263046
- gap_to_target: 252.263046
- target_achievement_ratio: -0.905%

# 2. 上一轮奖励函数代码（该轮得分: -2.263046）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack signals ---
    # crate-to-dock offsets (normalized by half-width 5.0, half-height 4.0)
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # crate velocity (world frame, normalized by 3.0)
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # crate heading
    crate_cos = obs[10]
    crate_sin = obs[11]

    # cart forward speed (normalized by 3.0)
    cart_fwd = obs[4]

    # contact flag
    contact = obs[14]

    # --- A. crate_to_dock_progress: improvement_delta on distance ---
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0
    r_progress = w_progress * progress

    # --- B. crate_settling_and_alignment: joint_condition_proxy ---
    # proximity factor: 1 when at dock, decays with distance
    prox = 1.0 / (1.0 + 8.0 * dist_new)
    # speed factor: 1 when still, decays with speed
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    # alignment factor: crate heading aligned with dock axis (assume dock axis = world x)
    # |cos(heading)| close to 1 means aligned with x-axis
    align = abs(crate_cos)
    # geometric mean of three continuous factors
    settle = (prox * speed_factor * align) ** (1.0 / 3.0)
    w_settle = 3.0
    r_settle = w_settle * settle

    # --- C. fragile_impact_penalty: hinge on relative speed during contact ---
    # relative speed proxy: cart forward speed vs crate speed
    rel_speed = abs(cart_fwd - crate_speed)
    # hinge: only penalize when contact AND relative speed exceeds threshold
    impact_threshold = 0.15
    impact_excess = max(0.0, rel_speed - impact_threshold)
    w_impact = 5.0
    r_impact = -w_impact * impact_excess * contact

    # --- D. out_of_bounds_penalty: hinge on cart position near boundary ---
    cart_x = obs[0]
    cart_y = obs[1]
    # normalized bounds [-1, 1]; penalize when |pos| > 0.85
    bound_threshold = 0.85
    cart_x_excess = max(0.0, abs(cart_x) - bound_threshold)
    cart_y_excess = max(0.0, abs(cart_y) - bound_threshold)
    w_bounds = 8.0
    r_bounds = -w_bounds * (cart_x_excess + cart_y_excess)

    # --- total ---
    total_reward = r_progress + r_settle + r_impact + r_bounds

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_settling_and_alignment": r_settle,
        "fragile_impact_penalty": r_impact,
        "out_of_bounds_penalty": r_bounds,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-2.263046, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-3.137673, -0.752075]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_settling_and_alignment | 605.967319 | 100.0% | 100.0% | 100.0% |
| crate_to_dock_progress | 0.000000 | 0.0% | 0.0% | 0.0% |
| fragile_impact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| out_of_bounds_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只**易碎**的方形货箱从仓库近侧推到隔墙另一侧的指定停靠区（dock）。主目标是让货箱**完整进入 dock 矩形、朝向与 dock 对齐（误差 < 30°）、速度接近静止（< 0.05 m/s），并连续保持 10 步**。次目标包括：避免货箱受到 3 次以上硬冲击（易碎约束）、避免货箱或小车越出仓库边界、在时间预算内完成。**不该混淆的目标**：单纯“接触 dock 区域”不算成功；单纯“把货箱推得离 dock 更近”也不等于完成（因为小车无法从后方减速货箱，货箱会因地面阻尼继续滑行，最终需要低速停稳）。因此这是一个**带阶段目标 + 软接触停靠 + 安全约束**的操控任务，而非纯导航或纯前进任务。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 所有分量裁剪到 [-2.0, 2.0]
- obs[0]: cart_x，小车 x 位置 / 仓库半宽（0=中线，+1=远墙），reward_usable: true
- obs[1]: cart_y，小车 y 位置 / 仓库半高（0=中线），reward_usable: true
- obs[2]: cart_cos_heading，小车朝向余弦，reward_usable: true
- obs[3]: cart_sin_heading，小车朝向正弦，reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身朝向速度 / 3.0 (m/s)，reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度 / 8.0 (rad/s)，reward_usable: true
- obs[6]: crate_rel_x_body，货箱相对小车在车体坐标系 x 分量 / 3.0 (m)，reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体坐标系 y 分量 / 3.0 (m)，reward_usable: true
- obs[8]: crate_vx，货箱世界系线速度 x / 3.0 (m/s)，reward_usable: true
- obs[9]: crate_vy，货箱世界系线速度 y / 3.0 (m/s)，reward_usable: true
- obs[10]: crate_cos_heading，货箱朝向余弦，reward_usable: true
- obs[11]: crate_sin_heading，货箱朝向正弦，reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到 dock 中心的有符号 x 偏移 / 仓库半宽，reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到 dock 中心的有符号 y 偏移 / 仓库半高，reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱当前是否接触（1.0/0.0），reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=远，1=接触），reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例 [0,1]，reward_usable: true（仅用于时间相关 shaping，需谨慎）

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [2]
- bounds: 每通道 [-1.0, 1.0]
- action[0] drive：沿小车朝向的纵向力指令；+1 前进，-1 倒车
- action[1] steer：转向力矩指令；+1 左转（逆时针），-1 右转
- 说明：无刹车通道；小车无法主动减速货箱，货箱仅靠地面阻尼减速。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `stable_steps >= 10`，即货箱完整在 dock 内、朝向误差 < 30°、速度 < 0.05 m/s，连续保持 10 步。
- failure-like termination: (a) 货箱中心越出仓库地板矩形；(b) 小车中心越出仓库地板矩形；(c) `hard_collision_count >= 3`（货箱受到 3 次以上硬冲击，单次硬冲击定义为 cart-crate 接触峰值法向冲量超过易碎阈值）。
- ambiguous termination: 无显式歧义终止；但“货箱进入 dock 但未满足朝向/速度/持续条件”不会终止，属于未完成状态。
- truncation: `elapsed_steps >= MAX_EPISODE_STEPS`，报告为 truncation，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被禁止读取）
- explicit_failure_flag_available: false（info 中 `termination_reason`、`hard_collision_count` 被禁止读取）
- allowed_info_fields: []（无任何允许的 info 字段）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

## 7. 可用于奖励函数的信号
- position:
  - 小车世界位置：由 obs[0]*5.0, obs[1]*4.0 恢复（仓库半宽 5.0，半高 4.0）
  - 货箱世界位置：将 (obs[6]*3.0, obs[7]*3.0) 按小车朝向旋转后加上小车位置
  - 货箱到 dock 偏移：obs[12]*5.0, obs[13]*4.0（直接可用）
- velocity:
  - 小车前向速度：obs[4]*3.0
  - 小车偏航率：obs[5]*8.0
  - 货箱世界速度：obs[8]*3.0, obs[9]*3.0；货箱轴向速率 = sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)
- orientation:
  - 小车朝向：atan2(obs[3], obs[2])
  - 货箱朝向：atan2(obs[11], obs[10])
  - 货箱朝向误差：需与 dock 朝向比较（dock 朝向未显式给出，需从任务语义推断或作为 derived_possible）
- contact:
  - cart_crate_contact：obs[14]（0/1）
  - 静态障碍接近度：obs[15], obs[16], obs[17]（隔墙/边界接近）
- action/engine:
  - action[0] drive, action[1] steer（可用于动作平滑/能耗 shaping，但任务未明确要求节能）
- other:
  - time_fraction：obs[18]
  - derived_possible（间接推断）：
    - 货箱越界：货箱世界位置超出仓库矩形（由 obs[6..7] + 小车位置恢复）
    - 小车越界：obs[0], obs[1] 超出 [-1,1] 合理范围
    - 硬冲击/易碎风险：cart_crate_contact=1 且小车-货箱相对速度较大（由 obs[4], obs[8], obs[9] 组合推断），**只能作为风险代理，不能精确复现 hard_collision_count**
    - 成功停靠：货箱到 dock 偏移接近 0、货箱速度接近 0、货箱朝向与 dock 对齐、且持续若干步（由 obs[12..13], obs[8..9], obs[10..11] 组合推断）

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crate_settling_and_alignment + crate_to_dock_progress + fragile_impact_penalty + out_of_bounds_penalty | -2.26 | -2.26 | 0.00 | 400.00 | crate_settling_and_alignment=1.510 crate_to_dock_progress=0.000 fragile_impact_penalty=-0.000 out_of_bounds_penalty=-0.005 | new_best |
