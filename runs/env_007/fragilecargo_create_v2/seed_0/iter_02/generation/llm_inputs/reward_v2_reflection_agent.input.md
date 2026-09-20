# 1. Search objective
- target_score: 250.000000
- current_score: 4.503913
- gap_to_target: 245.496087
- target_achievement_ratio: 1.802%

# 2. 上一轮奖励函数代码（该轮得分: 4.503913）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到坞的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱到坞距离（归一化尺度）
    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向误差（弧度），坞假设与坐标轴对齐
    crate_heading = (obs[11] / (1.0 + 1e-6)) if False else 0.0  # placeholder, replaced below
    # 用 atan2 计算朝向角，再取与 0 的夹角（坞对齐假设为 0 朝向）
    import_guard = 0.0  # no-op, keep function pure
    ch = obs[10]
    sh = obs[11]
    # 朝向角
    heading_angle = 0.0
    # 手动 atan2 近似：使用 cos/sin 直接构造对齐度，避免 atan2 依赖
    # 对齐度 = cos(heading)（坞朝向假设为 0，即 cos=1 表示对齐）
    align_cos = ch  # 范围 [-1, 1]，1 表示完全对齐

    # 接触标志
    contact = obs[14]

    # 小车前向速度（归一化）
    cart_v = obs[4]

    # ---------- 组件 A: 货箱向坞的进度（delta 形式，防悬停） ----------
    # 用 improvement_delta：距离减少为正
    progress = (dist - next_dist)  # 正=靠近
    # 低速门控：接近坞时抑制速度，防止高速滑过
    # 门控因子在 dist 小时衰减，但不阻断早期探索（dist 大时接近 1）
    # 使用线性衰减门：dist < 0.15 时开始衰减
    near_gate = 1.0
    if dist < 0.15:
        near_gate = max(0.2, dist / 0.15)  # 最低保留 0.2，避免完全阻断
    # 货箱速率门控：接近坞时若速度高，抑制进度奖励
    speed_gate = 1.0
    if dist < 0.15:
        # 速度越高，门控越低（但保留最低 0.2）
        speed_gate = max(0.2, 1.0 / (1.0 + 2.0 * crate_speed))
    progress_reward = 5.0 * progress * near_gate * speed_gate

    # ---------- 组件 B: 货箱朝向对齐（仅在接近坞时启用） ----------
    # 对齐度：align_cos 在 [-1,1]，映射到 [0,1]
    align_factor = (align_cos + 1.0) * 0.5  # 0=反向, 1=对齐
    # 仅在货箱接近坞时启用（dist < 0.3）
    align_gate = 0.0
    if dist < 0.3:
        align_gate = max(0.0, 1.0 - dist / 0.3)
    alignment_reward = 0.5 * align_factor * align_gate

    # ---------- 组件 C: 货箱在坞内近静止（settling，仅在近坞时启用） ----------
    # 速率惩罚：仅在货箱接近坞时启用，避免抑制必要推动
    settle_gate = 0.0
    if dist < 0.2:
        settle_gate = max(0.0, 1.0 - dist / 0.2)
    # 速率超过 0.05 m/s 时惩罚（hinge 形式）
    speed_excess = max(0.0, crate_speed - 0.05)
    settling_penalty = -1.0 * speed_excess * settle_gate

    # ---------- 组件 D: 边界/障碍安全（hinge 形式，轻量） ----------
    # 小车位置越界风险：obs[0], obs[1] 接近 ±1 时惩罚
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    # 小车边界 hinge：|x| > 0.85 或 |y| > 0.85 时惩罚
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    # 障碍接近度惩罚：sensor 接近 1 时轻罚
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.8:
        boundary_penalty -= 0.3 * (sensor_max - 0.8)

    # ---------- 组件 E: 接触冲击抑制（保守，仅在接触时） ----------
    # 接触时若货箱速度高，轻罚（间接推断硬碰撞风险）
    impact_penalty = 0.0
    if contact > 0.5:
        # 接触时货箱速率超过 1.0 m/s 时轻罚
        impact_excess = max(0.0, crate_speed - 1.0)
        impact_penalty = -0.3 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + alignment_reward
        + settling_penalty
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_dock_alignment": float(alignment_reward),
        "crate_settling": float(settling_penalty),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=4.503913, len=400.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[3.575489, 8.861561]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crate_dock_alignment | 101.329894 | 80.8% | 80.8% | 65.0% |
| crate_settling | -12.158012 | -9.7% | 9.7% | 14.6% |
| soft_contact_penalty | -8.430463 | -6.7% | 6.7% | 16.6% |
| crate_to_dock_progress | 3.535816 | 2.8% | 2.8% | 45.8% |
| boundary_avoidance | -0.002291 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只自由滑动的方形易碎货箱，从隔墙近侧推到隔墙远侧的矩形交付坞内，并让货箱在坞内**完全进入、朝向对齐、几乎静止**且持续一小段时间。主目标是“把货箱稳定交付到坞内”，次目标是“过程中不损坏货箱、不越界、不超时”。**不该混淆的目标**：单纯靠近坞、单纯接触坞、单纯把货箱推快、单纯让小车前进，都不等于完成任务；货箱在坞内高速滑过或朝向歪斜也不算成功。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 全部裁剪到 [-2.0, 2.0]
- obs[0]: cart_x，小车 x 位置 / 仓库半宽（0=中线，+1=远墙），reward_usable: true
- obs[1]: cart_y，小车 y 位置 / 仓库半高（0=中线），reward_usable: true
- obs[2]: cart_cos_heading，小车朝向余弦，reward_usable: true
- obs[3]: cart_sin_heading，小车朝向正弦，reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身朝向速度 / 3.0 (m/s)，reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度 / 8.0 (rad/s)，reward_usable: true
- obs[6]: crate_rel_x_body，货箱相对小车在车体系 x 分量 / 3.0 (m)，reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体系 y 分量 / 3.0 (m)，reward_usable: true
- obs[8]: crate_vx，货箱世界系线速度 x / 3.0 (m/s)，reward_usable: true
- obs[9]: crate_vy，货箱世界系线速度 y / 3.0 (m/s)，reward_usable: true
- obs[10]: crate_cos_heading，货箱朝向余弦，reward_usable: true
- obs[11]: crate_sin_heading，货箱朝向正弦，reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到坞中心的有符号 x 偏移 / 仓库半宽，reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到坞中心的有符号 y 偏移 / 仓库半高，reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱是否接触（1/0），reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=远，1=接触），reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例 [0,1]，reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [2]
- bounds: 每通道 [-1.0, 1.0]
- action[0] drive：沿小车朝向的纵向力指令；+1 前进，-1 倒车
- action[1] steer：转向力矩指令；+1 左转（逆时针），-1 右转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `docked_success` —— 货箱完全在坞内、朝向误差 < 30°、速度 < 0.05 m/s，且连续保持 10 个环境步。
- failure-like termination: `crate_out_of_bounds`（货箱中心离开仓库地面矩形）、`cart_out_of_bounds`（小车中心离开仓库地面矩形）、`crate_damaged`（硬碰撞计数 ≥ 3，硬碰撞定义为峰值法向冲量超过易碎阈值的车-箱接触）。
- ambiguous termination: 无显式歧义终止；但“货箱进入坞但未满足朝向/速度/持续条件”不会终止，会继续运行，属于未完成状态。
- truncation: `time_limit` —— 达到固定步数预算，报告为截断，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被列为 forbidden）
- explicit_failure_flag_available: false（`termination_reason`、`hard_collision_count` 等均 forbidden）
- allowed_info_fields: []（无任何允许读取的 info 字段）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

## 7. 可用于奖励函数的信号
- position: obs[0], obs[1]（小车位置）；obs[6], obs[7]（货箱相对小车车体系位置）；obs[12], obs[13]（货箱到坞的有符号偏移，可直接作为距离/方向信号）。货箱世界坐标可由 obs[0..3] 与 obs[6..7] 反推（derived_possible）。
- velocity: obs[4]（小车前向速度）；obs[5]（小车角速度）；obs[8], obs[9]（货箱世界系速度，可算货箱速率 sqrt((obs[8]*3)^2+(obs[9]*3)^2)）；obs[5] 与 obs[8..9] 组合可推断货箱是否近静止（derived_possible）。
- orientation: obs[2], obs[3]（小车朝向）；obs[10], obs[11]（货箱朝向，可算朝向误差 atan2(obs[11],obs[10])，与坞对齐程度 derived_possible）。
- contact: obs[14]（车-箱接触标志）；obs[15..17]（前/左/右静态障碍接近度，可推断是否接近隔墙/开口/边界，derived_possible）。
- action/engine: action[0]（drive）、action[1]（steer），可用于动作平滑/能耗类信号（但本任务未明确要求节能，属附属）。
- other: obs[18]（时间比例，可用于时间相关 shaping 或截断前提示，但需谨慎）；由 obs[12], obs[13] 可算货箱到坞距离（derived_possible）；由 obs[8..11] 可算货箱速率与朝向误差（derived_possible）；由 obs[15..17] 与位置可推断越界风险（derived_possible）。

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
| 1 | boundary_avoidance + crate_dock_alignment + crate_settling + crate_to_dock_progress + soft_contact_penalty | 4.50 | 4.50 | 0.00 | 400.00 | boundary_avoidance=-0.000 crate_dock_alignment=0.143 crate_settling=-0.017 crate_to_dock_progress=0.007 soft_contact_penalty=-0.009 | new_best |
