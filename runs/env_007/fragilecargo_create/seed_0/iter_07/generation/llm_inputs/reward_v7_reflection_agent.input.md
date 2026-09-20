# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 250.000000
- current_score: -6.735980
- gap_to_target: 256.735980
- target_achievement_ratio: -2.694%

# 2. 上一轮奖励函数代码（该轮得分: -6.735980）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack (仅使用环境声明的 obs 维度) ----------
    # 货箱到 dock 的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱世界速度
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向
    crate_cos = obs[10]
    crate_sin = obs[11]

    # 小车
    cart_fwd = obs[4]
    contact = obs[14]
    cart_x = obs[0]
    cart_y = obs[1]

    # ---------- A. MAIN: dock_approach_improvement (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    # 主信号：仅对"靠近"给正分，远离给负分（对称，避免刷分）
    w_progress = 25.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (joint_condition_proxy, 几何平均) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重压低到主信号的 ~0.3x 量级，且不随步数无界累积
    w_joint = 0.6
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. boundary_health_penalty (hinge) ----------
    bx = max(0.0, abs(cart_x) - 0.9)
    by = max(0.0, abs(cart_y) - 0.9)
    w_bounds = 6.0
    r_bounds = -w_bounds * (bx + by)

    # ---------- total ----------
    total_reward = r_progress + r_joint + r_impact + r_bounds

    components = {
        "dock_approach_improvement": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "boundary_health_penalty": r_bounds,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 400.00 | -2.26 | ✅ |
| 2 | 骨架变化: crate_settling_and_alignment + crate_to_dock_progr | — | 400.00 | 1.61 | ✅ |
| 3 | 骨架变化: crate_settling_and_alignment + crate_to_dock_progr | — | 400.00 | 3.86 | ✅ |
| 4 | 骨架变化: boundary_health_penalty + crate_to_dock_progress + | — | 400.00 | -1.54 | ➖ |
| 5 | 骨架变化: boundary_health_penalty + dock_approach_improvemen | — | 400.00 | -1.69 | ❌ |
| 6 | 骨架变化: boundary_health_penalty + dock_approach_improvemen | — | 399.00 | -6.74 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-6.735980, len=399.000000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-102.591639, -0.725095]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| joint_dock_completion | 87.712652 | 98.6% | 98.6% | 100.0% |
| boundary_health_penalty | -1.254612 | -1.4% | 1.4% | 1.6% |
| dock_approach_improvement | 0.000000 | 0.0% | 0.0% | 0.0% |
| fragile_impact_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

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

# 7. Formula Operator Library（完整版，用于 Level 3 重建）
# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |



# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crate_settling_and_alignment + crate_to_dock_progress + fragile_impact_penalty + out_of_bounds_penalty | -2.26 | -2.26 | 0.00 | 400.00 | crate_settling_and_alignment=1.510 crate_to_dock_progress=0.000 fragile_impact_penalty=-0.000 out_of_bounds_penalty=-0.005 | new_best |
| 2 | crate_settling_and_alignment + crate_to_dock_progress + fragile_impact_penalty + out_of_bounds_penalty | 1.61 | 1.61 | 0.00 | 400.00 | crate_settling_and_alignment=0.402 crate_to_dock_progress=0.012 fragile_impact_penalty=-0.001 out_of_bounds_penalty=-0.002 | new_best |
| 3 | crate_settling_and_alignment + crate_to_dock_progress + fragile_impact_penalty + out_of_bounds_penalty | 3.86 | 3.86 | 0.00 | 400.00 | crate_settling_and_alignment=0.054 crate_to_dock_progress=0.018 fragile_impact_penalty=-0.001 out_of_bounds_penalty=-0.002 | new_best |
| 4 | boundary_health_penalty + crate_to_dock_progress + fragile_impact_penalty + joint_dock_completion | -1.54 | 3.86 | -5.40 | 400.00 | boundary_health_penalty=-0.004 crate_to_dock_progress=0.000 fragile_impact_penalty=-0.000 joint_dock_completion=2.194 | unsolved_improving_continue_from_best |
| 5 | boundary_health_penalty + dock_approach_improvement + dock_completion_state + fragile_impact_penalty | -1.69 | 3.86 | -5.55 | 400.00 | boundary_health_penalty=-0.002 dock_approach_improvement=0.000 dock_completion_state=0.548 fragile_impact_penalty=-0.000 | no_meaningful_improvement |
| 6 | boundary_health_penalty + dock_approach_improvement + fragile_impact_penalty + joint_dock_completion | -6.74 | 3.86 | -10.60 | 399.00 | boundary_health_penalty=-0.002 dock_approach_improvement=0.000 fragile_impact_penalty=-0.000 joint_dock_completion=0.219 | no_meaningful_improvement |
