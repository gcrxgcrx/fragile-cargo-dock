# Environment card

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只**易碎**的方形货箱从仓库近侧推到隔墙另一侧的**交付泊位**内，并让货箱在泊位中**完全进入、朝向对齐、几乎静止**并持续一小段时间，才算完成。主目标是"把货箱安全送达并稳定停靠"；次目标是"轻柔操作（避免硬碰撞）"和"不越界"。**不该混淆的目标**：单纯靠近泊位、单纯接触泊位、单纯把货箱推快、单纯让小车前进——这些都不是成功条件，且由于小车无法从后方减速货箱，任何"一直推到最后一刻"的策略都会失败。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: medium
reason: 核心是"把物体（货箱）移动到指定位姿（泊位内、朝向对齐、静止）"，属于物体搬运/操控类，而非小车自身的导航或步态前进。虽然动作是轮式底盘驱动、没有夹爪，但成功判据完全由**货箱**的位置/朝向/速度决定，小车只是推动工具，因此归为 manipulation_grasping 最贴切。次目标（轻柔、不越界）是附属约束，不构成 multi_objective。

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
- obs[6]: crate_rel_x_body，货箱相对小车在车体系 x 分量 / 3.0 (m)，reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体系 y 分量 / 3.0 (m)，reward_usable: true
- obs[8]: crate_vx，货箱世界系线速度 x / 3.0 (m/s)，reward_usable: true
- obs[9]: crate_vy，货箱世界系线速度 y / 3.0 (m/s)，reward_usable: true
- obs[10]: crate_cos_heading，货箱朝向余弦，reward_usable: true
- obs[11]: crate_sin_heading，货箱朝向正弦，reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到泊位中心的有符号 x 偏移 / 仓库半宽，reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到泊位中心的有符号 y 偏移 / 仓库半高，reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱当前是否接触（1/0），reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=空，1=接触），reward_usable: true
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
- success-like termination: `docked_success` —— 货箱完全在泊位内、朝向误差 < 30°、速度 < 0.05 m/s，且连续保持 10 个环境步。成功时 episode 立即结束，之后不再累积奖励。
- failure-like termination: `crate_out_of_bounds`（货箱中心离开仓库地面矩形）、`cart_out_of_bounds`（小车中心离开仓库地面矩形）、`crate_damaged`（货箱累计 ≥3 次硬碰撞，硬碰撞定义为峰值法向冲量超过易碎阈值的车-箱接触）。
- ambiguous termination: 无显式歧义终止；但"货箱进入泊位但未满足朝向/静止/持续条件"不会终止，属于未完成状态。
- truncation: `time_limit` —— 达到固定步数预算，报告为 truncation，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被禁止读取）
- explicit_failure_flag_available: false（`termination_reason`、`hard_collision_count` 等被禁止）
- allowed_info_fields: []（无任何允许的 info 字段）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测向量，19 维）
- action（当前动作，2 维）
- next_obs（下一步观测向量，19 维）
- info 中明确允许的字段：**无**（allowed_info_fields 为空）
- training_progress：仅当 prompt 明确允许时才用；本环境未声明允许，默认不使用

禁止使用：
- original_reward（官方奖励被 mask）
- official_reward / 任何官方奖励项
- 未声明的 info 字段（全部 info 字段均被禁止）
- 未声明的 obs 切片（只能使用上述 19 维已声明含义）

## 7. 可用于奖励函数的信号
- position: obs[0], obs[1]（小车位置）；obs[6], obs[7]（货箱相对小车车体系位置）；obs[12], obs[13]（货箱到泊位的有符号偏移，**直接可用**）。货箱世界坐标可由 obs[6]*3.0, obs[7]*3.0 经小车朝向旋转后加小车位置 (obs[0]*5.0, obs[1]*4.0) 精确恢复（derived_possible）。
- velocity: obs[4]（小车前向速度）；obs[5]（小车偏航率）；obs[8], obs[9]（货箱世界系速度，货箱轴向速度 = sqrt((obs[8]*3.0)^2+(obs[9]*3.0)^2)，derived_possible）。
- orientation: obs[2], obs[3]（小车朝向）；obs[10], obs[11]（货箱朝向，货箱朝向误差 = atan2(obs[11], obs[10])，derived_possible）。
- contact: obs[14]（车-箱接触标志）。硬碰撞/冲量**不可直接读取**，但可通过接触标志 + 速度突变间接推断（derived_possible，可靠性低）。
- action/engine: action[0]（drive）、action[1]（steer），可用于动作平滑/能耗类信号。
- other: obs[15], obs[16], obs[17]（前/左/右障碍接近度，可用于避障/防撞信号）；obs[18]（时间比例，可用于时间相关 shaping，但需谨慎）。

## 8. 不确定或不可用的信号
- 硬碰撞次数、峰值法向冲量：info 被禁止，obs 无直接字段，只能通过 obs[14] 接触 + 速度突变间接推断，**不可靠**。
- 货箱是否"完全在泊位内"的官方布尔量：不可读，但可由 obs[12], obs[13] 与几何阈值精确重建（|obs[12]| ≤ 0.024 且 |obs[13]| ≤ 0.030）。
- 稳定步数 stable_steps：不可读，需自行在奖励函数内维护计数器（若允许状态）。
- 终止原因 termination_reason：不可读。
- 官方奖励分量：全部不可读。
- 货箱到泊位的真实距离（米）：info 被禁止，但可由 obs[12], obs[13] 乘以半宽/半高恢复（derived_possible）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: goal_approach_and_soft_contact
control_type: continuous
morphology:
  body_type: wheeled_cart_without_brake
  actuator_type: longitudinal_force_plus_steering_torque
  contact_structure: single_free_rigid_crate_pushed_by_contact_no_gripper
primary_objectives:
  - 将货箱推入泊位并使其完全进入（|obs[12]|<=0.024, |obs[13]|<=0.030）
  - 使货箱朝向对齐泊位（朝向误差 < 30°）
  - 使货箱在泊位内几乎静止（速度 < 0.05 m/s）并持续 10 步
secondary_objectives:
  - 轻柔操作，避免硬碰撞（累计 <3 次）
  - 不越界（小车与货箱均留在仓库地面内）
  - 在时间预算内完成
main_failure_risks:
  - 硬碰撞导致货箱损坏提前终止
  - 小车或货箱越界
  - 货箱滑过泊位（小车无法从后方减速）
  - 货箱进入泊位但朝向/速度不满足，无法稳定停靠
  - 时间耗尽（truncation）
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向泊位中心靠近，是任务的核心进度信号。
  why_required: 成功判据是货箱到达泊位，必须有信号引导货箱位置收敛到泊位。
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13]]
  risks: 若用 proximity（距离本身）作唯一主信号，货箱可能停在泊位附近但不进入，形成悬停陷阱；应优先用 delta(distance) 或 improvement。
- role_id: crate_docking_quality
  purpose: 在货箱接近泊位后，引导其满足"完全进入 + 朝向对齐 + 静止"的复合条件。
  why_required: 仅靠近不足以成功，必须满足几何与运动学条件。
  usable_signals: [obs[10], obs[11], obs[12], obs[13], obs[8], obs[9]]
  risks: 复合条件过严会导致信号稀疏；需分阶段或分项 shaping。

### 10.2 条件职责 conditional_roles
- role_id: soft_contact_penalty
  condition_to_use: 当需要抑制硬碰撞时使用；由于硬碰撞不可直接读取，只能用接触标志 + 速度突变间接推断，可靠性有限。
  usable_signals: [obs[14], obs[4], obs[8], obs[9]]
  risks: 间接推断可能误判正常推动为硬碰撞，导致惩罚噪声。
- role_id: crate_speed_penalty_near_dock
  condition_to_use: 当货箱接近泊位时，抑制其速度以促成静止停靠。
  usable_signals: [obs[8], obs[9], obs[12], obs[13]]
  risks: 过早惩罚速度会阻碍货箱到达泊位；应仅在接近泊位时启用。
- role_id: out_of_bounds_penalty
  condition_to_use: 当小车或货箱接近仓库边界时使用。
  usable_signals: [obs[0], obs[1], obs[6], obs[7], obs[12], obs[13]]
  risks: 边界位置需从 obs 恢复，存在缩放误差。
- role_id: action_smoothness
  condition_to_use: 仅当任务明确要求平滑/节能时使用；本任务未明确要求，属可选。
  usable_signals: [action[0], action[1]]
  risks: 可能抑制必要的推动动作。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: hard_collision_count_penalty
  reason: 硬碰撞次数与冲量在 info 中被禁止，obs 无直接字段，无法可靠获取。
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse]
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用。
  forbidden_or_missing_signals: [original_reward, official_reward_terms, component_returns]
- role_id: success_flag_bonus
  reason: is_success / termination_reason 被禁止，无法直接读取成功标志。
  forbidden_or_missing_signals: [is_success, termination_reason, cargo_inside_dock, stable_steps]
- role_id: cart_forward_velocity_reward
  reason: 小车前进速度本身不是任务目标，货箱才是；奖励小车速度会诱导小车空跑。
  forbidden_or_missing_signals: []

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 真实米制距离（可由 obs 恢复） | delta(distance), improvement, bounded_signal | 用 delta 避免悬停陷阱 |
| crate_docking_quality | obs[10], obs[11], obs[12], obs[13], obs[8], obs[9] | cargo_inside_dock, stable_steps | bounded_signal, hinge, quadratic_penalty | 分项 shaping：位置/朝向/速度 |
| soft_contact_penalty | obs[14], obs[4], obs[8], obs[9] | contact_impulse, hard_collision_count | hinge, bounded_signal | 间接推断，可靠性低 |
| crate_speed_penalty_near_dock | obs[8], obs[9], obs[12], obs[13] | cargo_speed | hinge, quadratic_penalty | 仅在接近泊位时启用 |
| out_of_bounds_penalty | obs[0], obs[1], obs[6], obs[7], obs[12], obs[13] | 无 | hinge, bounded_signal | 边界需从 obs 恢复 |
| action_smoothness | action[0], action[1] | action_energy | quadratic_penalty | 可选，任务未明确要求 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱悬停在泊位附近但不进入 | 训练日志中 obs[12]/obs[13] 收敛到非零小值，episode 以 truncation 结束 | 强化 delta(distance) 主信号，加入进入泊位的稀疏 bonus |
| 货箱滑过泊位 | 货箱速度在接近泊位时仍高，obs[12]/obs[13] 符号翻转 | 在接近泊位时加入速度抑制信号 |
| 硬碰撞导致提前终止 | episode 长度骤短，接触后速度突变频繁 | 加入软接触惩罚（间接推断），降低推动速度 |
| 小车或货箱越界 | obs[0]/obs[1] 或恢复的货箱坐标接近边界 | 加入边界 hinge 惩罚 |
| 小车空跑不推箱 | obs[14] 长期为 0，货箱位置不变 | 强化货箱进度信号，弱化小车自身运动信号 |
| 朝向不对齐无法停靠 | 货箱进入泊位但 obs[10]/obs[11] 朝向误差大 | 加入朝向对齐 shaping |
| 时间耗尽 | obs[18] 接近 1 且未成功 | 调整进度信号强度或加入时间相关 shaping |



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
    # ---- 从观测中提取信号 ----
    # 货箱到泊位的有符号偏移（归一化）
    crate_dock_x = next_obs[12]
    crate_dock_y = next_obs[13]
    prev_dock_x = obs[12]
    prev_dock_y = obs[13]

    # 货箱世界系速度（归一化，m/s / 3.0）
    crate_vx = next_obs[8]
    crate_vy = next_obs[9]
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    # 货箱朝向误差（弧度）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_error = (crate_sin * crate_sin + (1.0 - crate_cos) * (1.0 - crate_cos)) ** 0.5

    # 接触标志
    contact = next_obs[14]

    # 障碍接近度
    sensor_front = next_obs[15]
    sensor_left = next_obs[16]
    sensor_right = next_obs[17]

    # ---- 1. 货箱向泊位靠近的进度信号（delta distance）----
    prev_dist = (prev_dock_x * prev_dock_x + prev_dock_y * prev_dock_y) ** 0.5
    curr_dist = (crate_dock_x * crate_dock_x + crate_dock_y * crate_dock_y) ** 0.5
    progress = prev_dist - curr_dist
    crate_to_dock_progress = 2.0 * progress

    # ---- 2. 货箱停靠质量（联合条件软代理）----
    # 位置因子：完全进入泊位 |x|<=0.024, |y|<=0.030
    pos_x_factor = max(0.0, 1.0 - abs(crate_dock_x) / 0.024)
    pos_y_factor = max(0.0, 1.0 - abs(crate_dock_y) / 0.030)
    pos_factor = pos_x_factor * pos_y_factor

    # 朝向因子：误差 < 30° (0.5236 rad)
    heading_factor = max(0.0, 1.0 - heading_error / 0.5236)

    # 静止因子：速度 < 0.05 m/s 对应归一化 0.05/3.0 ≈ 0.01667
    speed_factor = max(0.0, 1.0 - crate_speed / 0.01667)

    # 几何平均，避免乘积塌缩
    docking_quality = (pos_factor * heading_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 3.0 * docking_quality

    # ---- 3. 接近泊位时的速度抑制 ----
    # 仅在货箱接近泊位时启用（距离归一化阈值 0.15）
    near_dock = max(0.0, 1.0 - curr_dist / 0.15)
    crate_speed_penalty_near_dock = -1.0 * near_dock * crate_speed

    # ---- 4. 软接触惩罚（间接推断硬碰撞）----
    # 接触时货箱速度过大视为硬碰撞风险
    if contact > 0.5:
        excess_speed = max(0.0, crate_speed - 0.02)
        soft_contact_penalty = -1.5 * excess_speed
    else:
        soft_contact_penalty = 0.0

    # ---- 5. 越界惩罚（hinge，边界附近生效）----
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    # 小车位置归一化，接近 ±1 时惩罚
    cart_edge = max(abs(cart_x), abs(cart_y))
    cart_bound_penalty = -2.0 * max(0.0, cart_edge - 0.85)

    # 货箱位置恢复（近似）：用货箱到泊位偏移 + 泊位位置估计
    # 货箱越界通过 crate_dock 偏移的极端值间接检测
    crate_edge = max(abs(crate_dock_x), abs(crate_dock_y))
    crate_bound_penalty = -2.0 * max(0.0, crate_edge - 0.9)

    out_of_bounds_penalty = cart_bound_penalty + crate_bound_penalty

    # ---- 6. 障碍接近惩罚（防止撞墙）----
    obstacle_penalty = -0.5 * (max(0.0, sensor_front - 0.8) +
                                max(0.0, sensor_left - 0.8) +
                                max(0.0, sensor_right - 0.8))

    # ---- 7. 动作平滑（轻量，避免抖动）----
    action_smoothness = -0.05 * (action[0] * action[0] + action[1] * action[1])

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_docking_quality": float(crate_docking_quality),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "obstacle_penalty": float(obstacle_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = sum(components.values())
    return (float(total_reward), components)
```

# Reward reflection of the current reward (native task score = -67.4027)
### Task score

- mean_eval_reward: -67.40267827660098
- mean_episode_length: 289.95
- eval episodes: 20
- termination breakdown: {'terminated': 13, 'truncated': 7}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | -8.11 | 166.7 |
| 33% | -8.12 | 163.7 |
| 50% | -7.97 | 166.1 |
| 67% | -7.88 | 169.4 |
| 83% | -8.20 | 169.6 |
| 100% | -7.97 | 164.7 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| obstacle_penalty | -5.936 | -0.377 | -9.595 | 0.000 | -2.719 | -0.000 | -2.691 | -5.072 | -2.731 | 0.000 | -1.853 |
| out_of_bounds_penalty | -0.039 | -20.473 | 0.000 | 0.000 | -3.675 | -5.655 | -0.000 | -2.645 | 0.000 | 0.000 | 0.000 |
| crate_to_dock_progress | -24.509 | -16.570 | -5.324 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -2.215 | -2.550 | 0.000 |
| action_smoothness | -0.289 | 0.000 | 0.005 | -3.404 | 0.000 | 0.000 | -5.450 | -2.427 | 0.000 | 0.000 | 0.000 |
| total_reward | -18.265 | -0.081 | 0.000 | -8.743 | -0.000 | 0.000 | -2.758 | -0.000 | 0.000 | 0.000 | 0.000 |
| soft_contact_penalty | 0.020 | -3.468 | -0.136 | 0.001 | -6.394 | -3.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| crate_speed_penalty_near_dock | 0.000 | 0.000 | -0.002 | -5.318 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -1.944 | -2.313 |
| crate_docking_quality | 0.000 | 0.024 | -15.052 | -0.022 | 0.000 | -2.642 | 0.000 | 0.000 | -4.946 | -4.495 | -4.166 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| soft_contact_penalty | 100.0% | 0.8% | 0.1% | 100.0% | 0.0% | 10.9% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| action_smoothness | 0.6% | 100.0% | 0.0% | 0.0% | 11.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_to_dock_progress | 0.9% | 100.0% | 100.0% | 11.7% | 0.0% | 0.0% | 10.1% | 0.0% | 0.0% | 5.8% | 4.0% |
| obstacle_penalty | 4.0% | 5.8% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 8.5% | 7.7% | 0.0% | 100.0% |
| out_of_bounds_penalty | 9.4% | 0.0% | 11.1% | 0.4% | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| crate_docking_quality | 0.0% | 5.9% | 1.5% | 0.1% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| total_reward | 0.0% | 0.0% | 0.3% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 0.0% | 100.0% | 0.0% |
| crate_speed_penalty_near_dock | 100.0% | 1.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| total_reward | 0.0000 | 0.0000 | -21.8863 | 0.7422 |
| action_smoothness | -0.0055 | 0.0055 | -27.3677 | -0.6822 |
| obstacle_penalty | -0.0420 | 0.0000 | -112.3730 | 0.0000 |
| crate_to_dock_progress | -3.1310 | 8.0420 | 0.0000 | 0.0000 |
| out_of_bounds_penalty | -4.8659 | 0.0420 | -0.0097 | 0.0000 |
| crate_speed_penalty_near_dock | -8.0420 | 0.0025 | -3.9479 | -0.6822 |
| crate_docking_quality | 0.0000 | 3.1310 | 0.0000 | 0.0000 |
| soft_contact_penalty | 0.0024 | 4.8659 | -97.0192 | 0.0000 |