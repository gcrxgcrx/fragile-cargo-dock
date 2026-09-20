# Environment card

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车需要把一只自由滑动的方形货箱从仓库近侧推到隔墙另一侧的指定停靠区（dock）。主目标是让货箱**完整进入 dock 矩形、朝向与 dock 对齐（误差 < 30°）、速度接近静止（< 0.05 m/s），并连续保持 10 步**才算成功。次目标包括：避免货箱受到 3 次以上硬冲击（易碎）、避免货箱或小车离开仓库地面、在时间预算内完成。**不该混淆的目标**：单纯“碰到 dock 区域”不算成功；单纯“把货箱推得离 dock 更近”也不等于完成，因为货箱会因惯性滑行，且最终必须低速静止对齐。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: medium
reason: 核心目标是“把物体（货箱）移动到指定位姿（dock 内、朝向对齐、静止）”，属于典型的物体搬运/操控到目标位姿任务，与 manipulation_grasping 的“物体到目标了吗？”核心问题一致。虽然载体是轮式小车而非机械臂，且没有夹爪（只能推），但任务本质仍是 staged manipulation：先接近货箱、再推动、最后在 dock 内低速对齐停稳。不是 navigation_goal_reaching（目标不是小车自身到达），不是 locomotion_continuous_control（不是持续前进通过地形），也不是 autonomous_driving_safety（安全约束是附属而非核心进度目标）。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 所有维度裁剪到 [-2.0, 2.0]
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
- obs[14]: cart_crate_contact，小车与货箱是否接触（1.0/0.0），reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=远，1=接触），reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例 [0,1]，reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [2]
- bounds: 每通道 [-1.0, 1.0]
- action[0]: drive，沿小车朝向的纵向力指令；+1 前进，-1 后退
- action[1]: steer，转向力矩指令；+1 左转（逆时针），-1 右转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `docked_success` —— 货箱完整在 dock 内、朝向误差 < 30°、速度 < 0.05 m/s，且连续保持 10 步。这是唯一成功终止。
- failure-like termination: `crate_out_of_bounds`（货箱中心离开仓库地面）、`cart_out_of_bounds`（小车中心离开仓库地面）、`crate_damaged`（硬冲击计数 ≥ 3）。
- ambiguous termination: 无显式 ambiguous 分支；但“货箱进入 dock 但未满足朝向/速度/持续条件”不会终止，属于未完成状态。
- truncation: `time_limit` —— 达到固定步数预算，报告为 truncation，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被明确禁止）
- explicit_failure_flag_available: false（info 中 `termination_reason`、`hard_collision_count` 等被禁止）
- allowed_info_fields: []（空，info 全部字段禁止用于奖励）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

**间接推断路径（derived_possible）**：
- 成功接近：obs[12]、obs[13] 同时趋近 0，且 obs[8]、obs[9] 速度趋近 0，obs[10]/obs[11] 朝向与 dock 对齐。
- 货箱出界：由 obs[6]、obs[7] 结合 obs[0..3] 反推货箱世界坐标，超出仓库矩形范围。
- 小车出界：obs[0]、obs[1] 超出合理范围（|cart_x|>1 或 |cart_y|>1）。
- 硬冲击：obs[14] 接触信号 + 速度突变（obs[4] 或 obs[8]/obs[9] 的剧烈变化）可间接推断，但**无法精确计数**，只能作为风险信号。
- 时间耗尽：obs[18] 接近 1.0。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（全部 19 维，按上述语义切片）
- action（2 维连续动作）
- next_obs（用于计算 delta / 变化量）
- info 中明确允许的字段：**无**（allowed_info_fields 为空）
- training_progress：prompt 未明确允许，**默认不使用**

禁止使用：
- original_reward（官方奖励被 mask）
- official_reward（同上）
- 未声明的 info 字段（全部 info 字段均禁止）
- 未声明的 obs 切片（只能使用 0–18 维）
- info 中的 is_success / termination_reason / hard_collision_count / cargo_inside_dock / stable_steps 等

## 7. 可用于奖励函数的信号
- position: obs[0], obs[1]（小车位置）；obs[6], obs[7]（货箱相对小车位置，可反推货箱世界坐标）；obs[12], obs[13]（货箱到 dock 偏移，**直接可用**）
- velocity: obs[4]（小车前向速度）；obs[5]（小车角速度）；obs[8], obs[9]（货箱世界系速度，可算货箱速率）
- orientation: obs[2], obs[3]（小车朝向）；obs[10], obs[11]（货箱朝向，可算朝向误差 atan2(obs[11], obs[10])）
- contact: obs[14]（小车-货箱接触标志）；obs[15], obs[16], obs[17]（前方/左/右障碍接近度，可推断隔墙与边界风险）
- action/engine: action[0]（drive）、action[1]（steer），可用于动作平滑/能耗（仅在明确要求时）
- other: obs[18]（时间比例，可用于时间相关 shaping 或 gate）
- derived_possible:
  - 货箱世界坐标 = 旋转 (obs[6]*3.0, obs[7]*3.0) 按小车朝向 + 小车位置 (obs[0]*5.0, obs[1]*4.0)
  - 货箱到 dock 距离 = sqrt((obs[12]*5.0)^2 + (obs[13]*4.0)^2)
  - 货箱速率 = sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)
  - 货箱朝向误差 = atan2(obs[11], obs[10])
  - 货箱出界 / 小车出界可由上述坐标与仓库矩形比较推断
  - 硬冲击风险可由 obs[14] 接触 + 速度突变间接推断（不可精确计数）

## 8. 不确定或不可用的信号
- info 中所有字段（is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps）——**全部禁止**
- 硬冲击的精确计数（只能间接推断风险，不能精确判定第 3 次）
- dock 矩形的精确边界（obs[12]/obs[13] 给出中心偏移，但矩形尺寸未显式给出，只能近似判断“接近中心”）
- 隔墙开口的精确位置（只能通过 obs[15..17] 障碍接近度间接感知）
- 官方奖励的任何分量（被 mask）
- 成功/失败的显式布尔标志（不存在）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation
control_type: continuous
morphology:
  body_type: wheeled_cart_top_down
  actuator_type: longitudinal_force_and_steering_torque
  contact_structure: rigid_body_push_contact_no_gripper
primary_objectives:
  - 将货箱完整推入 dock 矩形
  - 使货箱朝向与 dock 对齐（误差 < 30°）
  - 使货箱在 dock 内低速静止（< 0.05 m/s）并保持 10 步
secondary_objectives:
  - 避免货箱受到 3 次以上硬冲击（易碎）
  - 避免货箱或小车离开仓库地面
  - 在时间预算内完成
main_failure_risks:
  - 货箱被推过头滑出 dock（无刹车，惯性滑行）
  - 货箱朝向未对齐导致无法满足成功条件
  - 硬冲击累积导致货箱损坏
  - 货箱或小车越界
  - 时间耗尽（truncation）
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向 dock 中心靠近，提供连续进度信号
  why_required: 任务核心是把货箱送到 dock，必须有指向目标的稠密信号
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13]]
  risks: 若只用 proximity（距离值）会诱导货箱停在 dock 附近但不进入；应使用 delta（距离减少量）或结合终端条件
- role_id: crate_dock_alignment
  purpose: 促使货箱朝向与 dock 对齐
  why_required: 成功条件要求朝向误差 < 30°，仅位置到位不够
  usable_signals: [obs[10], obs[11], next_obs[10], next_obs[11]]
  risks: 朝向信号在货箱静止时无梯度，需与进度信号配合
- role_id: crate_settling
  purpose: 促使货箱在 dock 内低速静止
  why_required: 成功条件要求速度 < 0.05 m/s 并保持 10 步
  usable_signals: [obs[8], obs[9], next_obs[8], next_obs[9]]
  risks: 过早惩罚速度会阻止推动；应在货箱接近 dock 时才启用

### 10.2 条件职责 conditional_roles
- role_id: contact_gated_push
  purpose: 在接触时给予推动相关的正向信号，鼓励有效推动
  condition_to_use: 当货箱尚未进入 dock 且需要推动时
  usable_signals: [obs[14], obs[4], obs[6], obs[7]]
  risks: 接触信号为 0/1，可能产生稀疏梯度；需与进度信号结合
- role_id: impact_softness
  purpose: 抑制硬冲击，保护易碎货箱
  condition_to_use: 当检测到接触且速度突变较大时
  usable_signals: [obs[14], obs[4], obs[8], obs[9], next_obs[4], next_obs[8], next_obs[9]]
  risks: 无法精确计数硬冲击，只能近似；过度惩罚会阻止必要推动
- role_id: boundary_avoidance
  purpose: 避免货箱或小车越界
  condition_to_use: 当货箱或小车接近仓库边界时
  usable_signals: [obs[0], obs[1], obs[6], obs[7], obs[15], obs[16], obs[17]]
  risks: 边界精确位置未显式给出，需从坐标范围推断
- role_id: time_efficiency
  purpose: 鼓励在时间预算内完成
  condition_to_use: 仅当任务明确要求效率时（本任务未明确要求，慎用）
  usable_signals: [obs[18]]
  risks: 可能诱导冒险行为，导致硬冲击或越界

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用
  forbidden_or_missing_signals: [original_reward, official_reward, component_returns, official_reward_terms]
- role_id: explicit_success_bonus
  reason: info 中 is_success / termination_reason 被禁止，无法直接读取成功标志
  forbidden_or_missing_signals: [is_success, termination_reason, cargo_inside_dock, stable_steps]
- role_id: hard_collision_penalty
  reason: hard_collision_count 被禁止，无法精确计数硬冲击
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse]
- role_id: action_energy_penalty
  reason: action_energy 被禁止；且任务未明确要求节能，属于附属优化
  forbidden_or_missing_signals: [action_energy]
- role_id: stagnation_penalty
  reason: stagnation_steps 被禁止；且停滞惩罚可能干扰必要的低速对齐阶段
  forbidden_or_missing_signals: [stagnation_steps]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 无（距离可直接算） | delta(distance), bounded_signal | 用 delta 避免悬停陷阱；距离 = sqrt((obs[12]*5)^2+(obs[13]*4)^2) |
| crate_dock_alignment | obs[10], obs[11], next_obs[10], next_obs[11] | dock 朝向精确值 | bounded_signal, quadratic_penalty | 朝向误差 = atan2(obs[11], obs[10])；仅在接近 dock 时启用 |
| crate_settling | obs[8], obs[9], next_obs[8], next_obs[9] | 无 | bounded_signal, hinge_penalty | 速率 = sqrt((obs[8]*3)^2+(obs[9]*3)^2)；仅在货箱接近 dock 时启用 |
| contact_gated_push | obs[14], obs[4], obs[6], obs[7] | 无 | gate, dense_state_signal | 接触时 gate 开启，鼓励有效推动 |
| impact_softness | obs[14], obs[4], obs[8], obs[9], next_obs[4], next_obs[8], next_obs[9] | hard_collision_count, contact_impulse | hinge_penalty, bounded_signal | 用速度突变近似冲击强度；无法精确计数 |
| boundary_avoidance | obs[0], obs[1], obs[6], obs[7], obs[15], obs[16], obs[17] | 仓库精确边界 | hinge_penalty, gate | 从坐标范围推断边界；障碍接近度辅助 |
| time_efficiency | obs[18] | 无 | bounded_signal | 任务未明确要求效率，慎用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱被推过头滑出 dock | 训练日志中货箱进入 dock 后速度未降，obs[8]/obs[9] 在 dock 附近仍较大 | 在货箱接近 dock 时增强 settling 信号，或加入 dock 内速度惩罚 |
| 货箱朝向未对齐 | 货箱位置到位但 obs[10]/obs[11] 朝向误差持续 > 30° | 增强 alignment 信号，或在接近 dock 时加入朝向 shaping |
| 硬冲击累积导致失败 | 训练中 episode 频繁在接触后突然终止，obs[14] 接触时速度突变大 | 加入 impact_softness 信号，抑制高速接触 |
| 货箱或小车越界 | obs[0]/obs[1] 或反推的货箱坐标超出仓库范围 | 加入 boundary_avoidance 信号，或调整 gate |
| 时间耗尽（truncation） | obs[18] 接近 1.0 时任务未完成 | 检查进度信号是否足够稠密，或调整时间效率信号 |
| 货箱卡在隔墙开口 | 货箱位置长时间停滞在隔墙附近，obs[15..17] 障碍接近度高 | 加入开口导航辅助信号，或调整推动策略 |
| 小车无法有效推动货箱 | obs[14] 接触频繁但 obs[12]/obs[13] 无改善 | 检查 contact_gated_push 信号，或调整推动角度 |
| 货箱在 dock 附近振荡 | obs[12]/obs[13] 在 0 附近波动，obs[8]/obs[9] 反复变化 | 调整 settling 信号，避免过度惩罚导致振荡 |



# Expert reward context

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



# Current reward function
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 信号提取（仅使用声明的 obs 维度） ----
    cur_dx = obs[12] * 5.0
    cur_dy = obs[13] * 4.0
    nxt_dx = next_obs[12] * 5.0
    nxt_dy = next_obs[13] * 4.0

    cur_dist = (cur_dx * cur_dx + cur_dy * cur_dy) ** 0.5
    nxt_dist = (nxt_dx * nxt_dx + nxt_dy * nxt_dy) ** 0.5

    # 货箱速率 (m/s)
    cur_crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5
    nxt_crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐程度：|cos(theta)| 越接近 1 越对齐
    crate_heading = (next_obs[11] * next_obs[11] + next_obs[10] * next_obs[10]) ** 0.5
    if crate_heading < 1e-6:
        align_cos = 0.0
    else:
        align_cos = abs(next_obs[10] / crate_heading)
    angle_err = 1.0 - align_cos

    # 接触标志
    contact = next_obs[14]

    # 小车速度/角速度
    cart_speed = next_obs[4] * 3.0
    cart_yaw = next_obs[5] * 8.0

    # 时间比例
    time_frac = next_obs[18]
    if time_frac < 0.0:
        time_frac = 0.0
    if time_frac > 1.0:
        time_frac = 1.0

    # 障碍接近度
    front_prox = next_obs[15]

    # ---- 组件 1: crate_to_dock_progress（势能差，稠密） ----
    # 用距离减少量作为稠密进度信号；不再用接触硬门控，避免稀疏
    delta_dist = cur_dist - nxt_dist
    if delta_dist > 0.5:
        delta_dist = 0.5
    if delta_dist < -0.5:
        delta_dist = -0.5
    progress_reward = 8.0 * delta_dist

    # ---- 组件 2: contact_gated_push ----
    # 接触时，货箱速度朝向 dock 的分量越大越好
    if contact > 0.5 and nxt_dist > 0.2:
        if cur_dist > 1e-6:
            dir_x = -cur_dx / cur_dist
            dir_y = -cur_dy / cur_dist
        else:
            dir_x = 0.0
            dir_y = 0.0
        push_vx = next_obs[8] * 3.0
        push_vy = next_obs[9] * 3.0
        push_align = push_vx * dir_x + push_vy * dir_y
        if push_align < 0.0:
            push_align = 0.0
        if push_align > 1.5:
            push_align = 1.5
        push_reward = 1.5 * push_align
    else:
        push_reward = 0.0

    # ---- 组件 3: crate_dock_alignment ----
    # 货箱接近 dock 时，朝向对齐给奖励；门控放宽到 4.0m 并提高权重
    if nxt_dist < 4.0:
        align_gate = 1.0 - nxt_dist / 4.0
        alignment_reward = 3.0 * align_gate * align_cos
    else:
        alignment_reward = 0.0

    # ---- 组件 4: crate_settling ----
    # 货箱接近 dock 时，速度越低越好；门控放宽到 3.0m 并提高权重
    if nxt_dist < 3.0:
        settle_gate = 1.0 - nxt_dist / 3.0
        speed_factor = 1.0 / (1.0 + 4.0 * nxt_crate_speed)
        settling_reward = 3.0 * settle_gate * speed_factor
    else:
        settling_reward = 0.0

    # ---- 组件 5: joint_condition_proxy（仅在接近 dock 时才有意义） ----
    # 关键修复：pos_factor 用陡峭衰减，远离 dock 时接近 0，避免"活着刷分"
    # 门控：只有 nxt_dist < 2.0 时才计算，且权重大幅降低
    if nxt_dist < 2.0:
        pos_factor = 1.0 / (1.0 + 10.0 * nxt_dist)
        align_factor = align_cos
        vel_factor = 1.0 / (1.0 + 20.0 * nxt_crate_speed)
        joint_proxy = (pos_factor * align_factor * vel_factor) ** (1.0 / 3.0)
        joint_reward = 1.0 * joint_proxy
    else:
        joint_reward = 0.0

    # ---- 组件 6: impact_softness ----
    impact_penalty = 0.0
    if contact > 0.5:
        crate_speed_jump = nxt_crate_speed - cur_crate_speed
        if crate_speed_jump > 0.0:
            impact_penalty -= 0.5 * crate_speed_jump
        if nxt_crate_speed > 1.0:
            impact_penalty -= 0.3 * (nxt_crate_speed - 1.0)

    # ---- 组件 7: boundary_avoidance ----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    boundary_penalty = 0.0
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    if front_prox > 0.8:
        boundary_penalty -= 0.3 * (front_prox - 0.8)

    cart_cos = next_obs[2]
    cart_sin = next_obs[3]
    rel_bx = next_obs[6] * 3.0
    rel_by = next_obs[7] * 3.0
    crate_wx = cart_cos * rel_bx - cart_sin * rel_by + cart_x * 5.0
    crate_wy = cart_sin * rel_bx + cart_cos * rel_by + cart_y * 4.0
    if abs(crate_wx) > 4.3:
        boundary_penalty -= 0.5 * (abs(crate_wx) - 4.3)
    if abs(crate_wy) > 3.4:
        boundary_penalty -= 0.5 * (abs(crate_wy) - 3.4)

    # ---- 组件 8: action_smoothness ----
    drive = action[0]
    steer = action[1]
    action_penalty = -0.02 * (drive * drive + steer * steer)

    # ---- 总奖励 ----
    components = {}
    components["crate_to_dock_progress"] = float(progress_reward)
    components["contact_gated_push"] = float(push_reward)
    components["crate_dock_alignment"] = float(alignment_reward)
    components["crate_settling"] = float(settling_reward)
    components["joint_condition_proxy"] = float(joint_reward)
    components["impact_softness"] = float(impact_penalty)
    components["boundary_avoidance"] = float(boundary_penalty)
    components["action_smoothness"] = float(action_penalty)

    total_reward = 0.0
    for key in components:
        total_reward += components[key]

    return float(total_reward), components
```

# Reward reflection of the current reward (native task score = 5.1258)
### Task score

- mean_eval_reward: 5.125764488424242
- mean_episode_length: 400
- eval episodes: 20
- termination breakdown: {'terminated': 0, 'truncated': 20}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 1335.16 | 398.6 |
| 33% | 1318.14 | 398.2 |
| 50% | 1331.73 | 397.2 |
| 67% | 1315.75 | 399.0 |
| 83% | 1333.63 | 398.9 |
| 100% | 1323.50 | 398.4 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 234.190 | 1052.237 | 1371.347 | 1549.070 | 1501.254 | 1515.079 | 1449.495 | 1457.963 | 1500.361 | 1620.024 | 1661.217 |
| crate_dock_alignment | 126.767 | 502.419 | 639.855 | 703.846 | 677.661 | 681.309 | 671.236 | 677.905 | 691.637 | 737.369 | 764.942 |
| crate_settling | 63.863 | 337.527 | 469.166 | 555.481 | 542.143 | 556.218 | 505.934 | 506.866 | 528.753 | 586.533 | 599.655 |
| contact_gated_push | 33.538 | 126.462 | 146.648 | 150.204 | 148.199 | 138.546 | 152.128 | 152.721 | 153.195 | 149.180 | 144.967 |
| joint_condition_proxy | 12.937 | 76.083 | 105.739 | 128.350 | 123.818 | 129.555 | 114.586 | 114.688 | 120.580 | 137.695 | 142.132 |
| crate_to_dock_progress | 6.283 | 21.640 | 25.063 | 27.531 | 27.060 | 27.162 | 25.752 | 26.046 | 26.555 | 28.484 | 28.570 |
| impact_softness | -1.033 | -4.445 | -7.703 | -8.787 | -9.393 | -9.194 | -11.070 | -11.120 | -11.059 | -10.398 | -10.230 |
| action_smoothness | -7.550 | -7.145 | -7.245 | -7.469 | -8.158 | -8.481 | -8.512 | -8.980 | -9.054 | -8.775 | -8.820 |
| boundary_avoidance | -0.615 | -0.305 | -0.175 | -0.085 | -0.077 | -0.035 | -0.558 | -0.163 | -0.246 | -0.063 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| crate_dock_alignment | 59.9% | 85.5% | 88.1% | 89.4% | 89.1% | 89.3% | 89.4% | 89.2% | 89.8% | 89.8% | 91.5% |
| crate_settling | 20.4% | 63.1% | 71.7% | 74.4% | 74.7% | 71.9% | 74.4% | 74.6% | 75.7% | 76.3% | 79.4% |
| contact_gated_push | 11.2% | 25.2% | 23.9% | 21.7% | 21.1% | 20.1% | 21.6% | 21.1% | 21.5% | 20.6% | 19.7% |
| joint_condition_proxy | 10.9% | 50.6% | 62.4% | 66.4% | 65.9% | 64.0% | 65.7% | 66.6% | 67.9% | 69.6% | 71.8% |
| crate_to_dock_progress | 33.7% | 55.0% | 51.3% | 46.2% | 46.5% | 44.7% | 46.5% | 46.7% | 47.9% | 45.8% | 46.3% |
| impact_softness | 7.8% | 21.2% | 22.1% | 20.4% | 20.4% | 19.1% | 21.9% | 21.5% | 21.8% | 20.1% | 19.7% |
| action_smoothness | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| boundary_avoidance | 1.6% | 0.7% | 0.4% | 0.1% | 0.1% | 0.2% | 0.8% | 0.3% | 0.4% | 0.2% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| action_smoothness | -8.1400 | 8.1400 | -13.4422 | -4.0032 |
| boundary_avoidance | -0.2316 | 0.2316 | -65.9544 | 0.0000 |
| contact_gated_push | 135.1311 | 135.1311 | 0.0000 | 199.7485 |
| crate_dock_alignment | 611.5689 | 611.5689 | 0.0000 | 898.7208 |
| crate_settling | 465.7181 | 465.7181 | 0.0000 | 758.8111 |
| crate_to_dock_progress | 24.1736 | 24.3395 | -26.6286 | 35.6005 |
| impact_softness | -8.4284 | 8.4284 | -20.4832 | 0.0000 |
| joint_condition_proxy | 106.5267 | 106.5267 | 0.0000 | 193.9943 |
| total_reward | 1326.3183 | 1326.8918 | -67.7695 | 2002.4486 |