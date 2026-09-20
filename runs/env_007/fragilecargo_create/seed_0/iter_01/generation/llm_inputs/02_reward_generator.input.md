# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只**易碎**的方形货箱从仓库近侧推到隔墙另一侧的指定停靠区（dock）。主目标是让货箱**完整进入 dock 矩形、朝向与 dock 对齐（误差 < 30°）、速度接近静止（< 0.05 m/s），并连续保持 10 步**。次目标包括：避免货箱受到 3 次以上硬冲击（易碎约束）、避免货箱或小车越出仓库边界、在时间预算内完成。**不该混淆的目标**：单纯“接触 dock 区域”不算成功；单纯“把货箱推得离 dock 更近”也不等于完成（因为小车无法从后方减速货箱，货箱会因地面阻尼继续滑行，最终需要低速停稳）。因此这是一个**带阶段目标 + 软接触停靠 + 安全约束**的操控任务，而非纯导航或纯前进任务。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: medium
reason: 核心目标是“把物体（货箱）移动到指定位姿（dock 内、朝向对齐、静止）”，符合 manipulation_grasping 的“物体到目标了吗？”核心问题。虽然动作是轮式小车推箱而非机械臂抓取，但任务本质是**物体位姿操控 + 阶段目标（接近→推入→停稳）**，且存在易碎/硬冲击约束。未选 navigation_goal_reaching，因为成功判据不是“agent 到达某点”，而是“被操控物体到达并稳定在目标位姿”；未选 locomotion_continuous_control，因为没有持续前进通过地形的语义；未选 multi_objective_task，因为“易碎/不越界”是安全约束而非与主目标权重相当的第二核心目标。

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

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（19 维观测向量，按第 3 节索引使用）
- action（2 维连续动作）
- next_obs（下一步观测，用于计算 delta / 变化量）
- info 中明确允许的字段：**无**（allowed_info_fields 为空）
- training_progress：仅当 prompt 明确允许时才用；本环境未声明允许，默认不使用

禁止使用：
- original_reward（官方奖励被 mask）
- official_reward
- 未声明的 info 字段（见 5.2 禁用列表）
- 未声明的 obs 切片（只能使用 obs[0..18] 的语义，不得越界或臆造维度）

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

## 8. 不确定或不可用的信号
- 官方奖励 masked_reward：不可用
- info 中所有诊断字段：不可用（is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps）
- 精确的硬冲击计数与峰值冲量：不可用，只能用接触 + 相对速度做风险代理
- dock 的精确朝向：未在 obs 中显式给出，需从任务语义推断（假设 dock 朝向与仓库轴对齐），标注为 derived_possible
- 精确的“稳定步数”计数器：不可用，需自行在 reward 内部维护状态或仅用瞬时量近似
- 货箱是否“完整在 dock 内”：不可用，需由货箱位置 + dock 矩形尺寸推断（dock 尺寸未显式给出，标注为 uncertain）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: goal_approach_and_soft_contact
control_type: continuous
morphology:
  body_type: wheeled_cart_without_brake
  actuator_type: longitudinal_force_plus_steering_torque
  contact_structure: cart_pushes_freely_moving_fragile_crate_via_rigid_contact
primary_objectives:
  - 将货箱完整推入 dock 矩形
  - 使货箱朝向与 dock 对齐（误差 < 30°）
  - 使货箱速度接近静止（< 0.05 m/s）并连续保持 10 步
secondary_objectives:
  - 避免货箱受到 3 次以上硬冲击（易碎约束）
  - 避免货箱或小车越出仓库边界
  - 在时间预算内完成
main_failure_risks:
  - 硬冲击导致货箱损坏（3 次即失败）
  - 货箱或小车越界
  - 货箱滑过 dock 无法停稳（小车无刹车，无法从后方减速）
  - 货箱朝向未对齐导致无法满足成功条件
  - 时间耗尽（truncation，非成功）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向 dock 中心靠近，提供接近阶段的主信号
  why_required: 任务核心是把货箱移动到 dock，必须有指向目标的进度信号
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13]]
  risks: 若只用 proximity（距离值）会允许货箱停在 dock 附近不完成；应使用 delta(distance) 或 improvement 形式
- role_id: crate_settling_and_alignment
  purpose: 在货箱接近 dock 后，奖励货箱低速、朝向对齐、位于 dock 内
  why_required: 成功条件要求货箱静止且朝向对齐，仅靠近不够
  usable_signals: [obs[8], obs[9], obs[10], obs[11], obs[12], obs[13]]
  risks: 若过早给强停稳奖励，agent 可能在远离 dock 处就减速；需与进度信号配合，或作为接近后的条件职责

### 10.2 条件职责 conditional_roles
- role_id: fragile_impact_penalty
  condition_to_use: 当 cart_crate_contact=1 且小车-货箱相对速度较大时（硬冲击风险代理）
  usable_signals: [obs[14], obs[4], obs[8], obs[9]]
  risks: 无法精确复现 hard_collision_count，只能做风险代理；过度惩罚可能抑制必要的推动接触
- role_id: out_of_bounds_penalty
  condition_to_use: 当货箱或小车位置接近/超出仓库矩形时
  usable_signals: [obs[0], obs[1], obs[6], obs[7], obs[2], obs[3]]
  risks: 边界尺寸需从任务语义推断（半宽 5.0，半高 4.0），推断错误会导致误罚
- role_id: action_smoothness
  condition_to_use: 仅当任务明确要求平滑/节能时；本任务未明确要求，默认不加
  usable_signals: [action[0], action[1]]
  risks: 可能抑制必要的推动动作，属于附属优化
- role_id: time_penalty
  condition_to_use: 仅当需要鼓励尽快完成时；本任务未明确要求速度
  usable_signals: [obs[18]]
  risks: 可能诱导冒险行为，增加硬冲击风险

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用
  forbidden_or_missing_signals: [original_reward, official_reward, component_returns, official_reward_terms]
- role_id: info_based_success_bonus
  reason: info 中 is_success / cargo_inside_dock / stable_steps 等被禁止读取
  forbidden_or_missing_signals: [is_success, cargo_inside_dock, stable_steps, termination_reason]
- role_id: precise_hard_collision_penalty
  reason: hard_collision_count 与 contact_impulse 被禁止读取，无法精确复现
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse]
- role_id: cart_forward_velocity_reward
  reason: 本任务不是前进通过地形，小车速度本身不是目标；奖励小车速度会诱导乱冲，增加硬冲击风险
  forbidden_or_missing_signals: [obs[4]]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 精确 dock 尺寸 | delta(distance), improvement, bounded_signal | 用 delta 避免悬停陷阱；距离可由 obs[12..13] 直接算 |
| crate_settling_and_alignment | obs[8], obs[9], obs[10], obs[11], obs[12], obs[13] | dock 精确朝向、稳定步数计数器 | bounded_signal, hinge_penalty, quadratic_penalty | 朝向误差需与 dock 朝向比较，dock 朝向为 derived_possible |
| fragile_impact_penalty | obs[14], obs[4], obs[8], obs[9] | hard_collision_count, contact_impulse | hinge_penalty, bounded_signal | 仅风险代理，不能精确复现硬冲击计数 |
| out_of_bounds_penalty | obs[0], obs[1], obs[6], obs[7], obs[2], obs[3] | 精确仓库边界（需推断） | hinge_penalty, bounded_signal | 边界尺寸从任务语义推断 |
| action_smoothness | action[0], action[1] | 无 | quadratic_penalty | 默认不加，任务未要求 |
| time_penalty | obs[18] | 无 | linear_penalty | 默认不加，任务未要求速度 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱被推过 dock 无法停稳 | 货箱速度长期 > 0.05，obs[12..13] 在 0 附近震荡后远离 | 增加接近阶段的减速 shaping，或在接近 dock 时降低推动力度 |
| 硬冲击导致提前失败 | episode 在货箱未到 dock 时突然终止，且终止前 obs[14]=1 且相对速度大 | 增加 fragile_impact_penalty，抑制高速接触 |
| 货箱或小车越界 | episode 突然终止，终止前 obs[0]/obs[1] 或货箱位置接近边界 | 增加 out_of_bounds_penalty，或调整边界推断 |
| 货箱朝向未对齐 | 货箱在 dock 内但 obs[10..11] 与 dock 朝向偏差大，无法满足成功条件 | 增加朝向对齐 shaping，或引导小车从正确角度推箱 |
| 货箱卡在隔墙开口 | 货箱位置长期停滞在隔墙附近，obs[15..17] 接近 1 | 增加通过开口的引导信号，或调整接近路径 |
| 时间耗尽（truncation） | episode 以 truncation 结束，obs[18] 接近 1 | 增加时间惩罚或提高进度信号强度 |
| 主信号被悬停收割 | 货箱停在 dock 附近但未进入，reward 仍为正 | 改用 delta(distance) 而非 proximity，避免停在中间状态得分 |
| 动作震荡 | action[0]/action[1] 高频大幅变化 | 增加 action_smoothness（仅在确认震荡时） |



# expert_reward_context.md

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

