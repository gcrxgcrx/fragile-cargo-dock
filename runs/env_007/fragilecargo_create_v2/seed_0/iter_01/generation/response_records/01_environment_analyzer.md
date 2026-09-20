# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只自由滑动的方形易碎货箱，从隔墙近侧推到隔墙远侧的矩形交付坞内，并让货箱在坞内**完全进入、朝向对齐、几乎静止**且持续一小段时间。主目标是“把货箱稳定交付到坞内”，次目标是“过程中不损坏货箱、不越界、不超时”。**不该混淆的目标**：单纯靠近坞、单纯接触坞、单纯把货箱推快、单纯让小车前进，都不等于完成任务；货箱在坞内高速滑过或朝向歪斜也不算成功。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: medium
reason: 核心是“把物体操控到指定位姿并稳定停住”，属于对物体的位姿操控（无夹爪，靠推挤接触实现），而非单纯导航或持续前进。虽然载体是轮式小车，但成功判据完全落在货箱的位姿与速度上（完全入坞 + 朝向对齐 + 近静止 + 持续 10 步），因此归为 manipulation_grasping 更贴切。次目标（不损坏、不越界、不超时）是约束而非并列主目标，故不选 multi_objective_task。

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
- official_reward
- 未声明的 info 字段（全部 info 字段均 forbidden）
- 未声明的 obs 切片（只能使用上述 19 维已声明含义）

## 7. 可用于奖励函数的信号
- position: obs[0], obs[1]（小车位置）；obs[6], obs[7]（货箱相对小车车体系位置）；obs[12], obs[13]（货箱到坞的有符号偏移，可直接作为距离/方向信号）。货箱世界坐标可由 obs[0..3] 与 obs[6..7] 反推（derived_possible）。
- velocity: obs[4]（小车前向速度）；obs[5]（小车角速度）；obs[8], obs[9]（货箱世界系速度，可算货箱速率 sqrt((obs[8]*3)^2+(obs[9]*3)^2)）；obs[5] 与 obs[8..9] 组合可推断货箱是否近静止（derived_possible）。
- orientation: obs[2], obs[3]（小车朝向）；obs[10], obs[11]（货箱朝向，可算朝向误差 atan2(obs[11],obs[10])，与坞对齐程度 derived_possible）。
- contact: obs[14]（车-箱接触标志）；obs[15..17]（前/左/右静态障碍接近度，可推断是否接近隔墙/开口/边界，derived_possible）。
- action/engine: action[0]（drive）、action[1]（steer），可用于动作平滑/能耗类信号（但本任务未明确要求节能，属附属）。
- other: obs[18]（时间比例，可用于时间相关 shaping 或截断前提示，但需谨慎）；由 obs[12], obs[13] 可算货箱到坞距离（derived_possible）；由 obs[8..11] 可算货箱速率与朝向误差（derived_possible）；由 obs[15..17] 与位置可推断越界风险（derived_possible）。

## 8. 不确定或不可用的信号
- 硬碰撞计数、峰值冲量、易碎阈值触发：info 中 forbidden，obs 中无直接字段；只能通过 obs[14] 接触标志 + 速度突变间接猜测，**不可靠**，不应作为主惩罚信号。
- 是否“完全在坞内”：info 中 `cargo_inside_dock` forbidden；只能由 obs[12], obs[13] 的偏移量近似判断（需结合坞尺寸，但坞尺寸未在 obs 中显式给出，属 uncertain）。
- 稳定步计数 `stable_steps`：forbidden，不可用。
- 终止原因 `termination_reason`：forbidden，不可用。
- 官方奖励各分量：forbidden，不可用。
- 货箱是否越界/小车是否越界：无显式标志，只能由 obs[0], obs[1] 与 obs[6..9] 反推位置是否超出合理范围（derived_possible，但边界值未显式给出，属 uncertain）。
- 隔墙开口的精确几何：obs 只给接近度传感器，无法精确知道开口位置，属 uncertain。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: goal_approach_and_soft_contact
control_type: continuous
morphology:
  body_type: wheeled_cart_without_brake
  actuator_type: longitudinal_force_plus_steering_torque
  contact_structure: free_sliding_fragile_crate_pushed_by_contact_no_gripper
primary_objectives:
  - 将货箱完全推入交付坞矩形内
  - 使货箱朝向与坞对齐（误差 < 30°）
  - 使货箱在坞内近静止（速度 < 0.05 m/s）并持续 10 步
secondary_objectives:
  - 避免货箱受到硬碰撞（易碎，3 次即失败）
  - 避免货箱或小车越出仓库地面
  - 在时间预算内完成
main_failure_risks:
  - 推箱过猛导致硬碰撞累积到 3 次
  - 货箱滑出仓库边界
  - 小车滑出仓库边界
  - 货箱进入坞但速度过高/朝向歪斜，无法满足稳定条件
  - 小车无刹车，货箱滑过坞后无法从后方减速
  - 隔墙开口狭窄，推箱路径易撞墙
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向坞中心靠近，提供连续可学的主信号。
  why_required: 任务核心是“把货箱送到坞”，必须有指向坞的进度信号，否则 agent 无方向。
  usable_signals: [obs[12], obs[13]]（货箱到坞有符号偏移，可算距离）；next_obs[12], next_obs[13] 用于 delta。
  risks: 若只用 proximity（距离本身）会诱导货箱停在坞附近但不入坞；应优先用 delta(distance) 或 improvement 形式。
- role_id: crate_dock_alignment
  purpose: 促使货箱朝向与坞对齐。
  why_required: 成功条件明确要求朝向误差 < 30°，是主目标的一部分。
  usable_signals: [obs[10], obs[11]]（货箱朝向，可算朝向误差）。
  risks: 朝向误差在货箱未接近坞时无意义，应作为条件职责或与进度耦合。
- role_id: crate_settling
  purpose: 促使货箱在坞内近静止。
  why_required: 成功条件要求速度 < 0.05 m/s 并持续 10 步；无刹车特性使“减速”成为关键难点。
  usable_signals: [obs[8], obs[9]]（货箱速度，可算速率）。
  risks: 若在货箱远离坞时就惩罚速度，会抑制必要的推动；应仅在货箱接近/进入坞时启用。

### 10.2 条件职责 conditional_roles
- role_id: soft_contact_penalty
  condition_to_use: 当 obs[14] 显示车-箱接触且货箱速度/相对速度出现突变时，才考虑加入温和惩罚；仅在能可靠推断硬碰撞时使用。
  usable_signals: [obs[14], obs[8], obs[9], obs[4]]（接触 + 速度变化）。
  risks: 硬碰撞计数与冲量在 info 中 forbidden，obs 无直接冲量信号，误判风险高；应保守使用或仅用 hinge 形式。
- role_id: boundary_avoidance
  condition_to_use: 当由 obs[0], obs[1] 或反推的货箱位置接近仓库边界时启用。
  usable_signals: [obs[0], obs[1], obs[6], obs[7], obs[15], obs[16], obs[17]]。
  risks: 边界精确值未显式给出，属 derived_possible，阈值需谨慎设定。
- role_id: time_efficiency
  condition_to_use: 仅当任务明确要求“尽快完成”时才加入；本任务描述未强调速度，属附属。
  usable_signals: [obs[18]]。
  risks: 可能诱导 agent 冒险推快，增加硬碰撞风险。
- role_id: action_smoothness
  condition_to_use: 仅当观察到动作抖动导致货箱失控时才加入。
  usable_signals: [action[0], action[1]]。
  risks: 本任务未要求节能/平滑，属附属优化，不应默认加入。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: hard_collision_penalty_direct
  reason: 硬碰撞计数与冲量在 info 中 forbidden，obs 无直接信号，无法可靠计算。
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse, termination_reason]
- role_id: success_bonus_from_info
  reason: info 中 is_success / cargo_inside_dock / stable_steps 均 forbidden，不能直接读取成功标志。
  forbidden_or_missing_signals: [is_success, cargo_inside_dock, stable_steps]
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用 original_reward 或任何官方分量。
  forbidden_or_missing_signals: [original_reward, official_reward_terms, component_returns]
- role_id: cart_forward_velocity_as_main
  reason: 小车前进速度本身不是任务目标；货箱才是被操控对象，用小车速度作主信号会偏离目标。
  forbidden_or_missing_signals: []
- role_id: energy_penalty_default
  reason: 任务未要求节能，属附属优化，默认不应加入。
  forbidden_or_missing_signals: [action_energy]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 坞尺寸精确值 | delta(distance), improvement, bounded_signal | 用 delta 避免悬停陷阱；距离可由偏移量平方和开方得到 |
| crate_dock_alignment | obs[10], obs[11] | 坞朝向精确值（假设与坐标轴对齐） | bounded_signal, quadratic_penalty | 朝向误差 = atan2(obs[11],obs[10])；仅在接近坞时启用 |
| crate_settling | obs[8], obs[9] | 无 | bounded_signal, hinge_penalty | 速率 = sqrt((obs[8]*3)^2+(obs[9]*3)^2)；仅在坞内/近坞时启用 |
| soft_contact_penalty | obs[14], obs[4], obs[8], obs[9] | contact_impulse, hard_collision_count | hinge_penalty | 仅间接推断，保守使用 |
| boundary_avoidance | obs[0], obs[1], obs[6], obs[7], obs[15..17] | 精确边界值 | hinge_penalty, bounded_signal | derived_possible，阈值需谨慎 |
| time_efficiency | obs[18] | 无 | linear_penalty | 附属，默认不加 |
| action_smoothness | action[0], action[1] | 无 | quadratic_penalty | 附属，默认不加 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱停在坞附近但不入坞（悬停） | 训练日志中 obs[12], obs[13] 距离小但 episode 未成功终止；货箱速度长期接近 0 但不在坞内 | 主信号改用 delta(distance) 而非 proximity；加入入坞条件触发的 settling 信号 |
| 货箱高速滑过坞 | 货箱速率在坞附近仍高；episode 未成功终止 | 在货箱接近坞时启用 settling 信号；引导 agent 提前松力 |
| 硬碰撞累积导致失败 | episode 早期终止且伴随 obs[14] 频繁为 1 与货箱速度突变 | 加入温和的接触/速度突变 hinge 惩罚；降低推动力度 |
| 货箱或小车越界 | episode 终止时 obs[0], obs[1] 或反推货箱位置接近/超出边界 | 加入边界 hinge 惩罚；调整接近度传感器权重 |
| 撞隔墙/卡在开口 | obs[15..17] 长期高值；货箱进度停滞 | 加入障碍接近度惩罚或引导绕行；检查开口通过策略 |
| 动作抖动导致货箱失控 | action[0], action[1] 高频大幅变化；货箱轨迹不稳 | 加入动作平滑惩罚（条件性） |
| 时间耗尽截断 | obs[18] 接近 1 且未成功终止 | 检查进度信号是否足够密集；必要时加入时间相关 shaping |
| 主信号被悬停收割 | 训练回报高但成功率低 | 审查主信号是否允许静止得分；改用 delta 或终端事件 |
