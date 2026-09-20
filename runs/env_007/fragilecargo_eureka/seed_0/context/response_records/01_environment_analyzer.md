# Response Record

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
