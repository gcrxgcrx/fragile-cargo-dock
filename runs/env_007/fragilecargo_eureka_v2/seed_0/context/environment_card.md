# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只自由滑动的方形易碎货箱，从隔墙近侧推到隔墙远侧的指定停靠区（dock）内。主目标是**把货箱完整送入 dock 矩形、货箱朝向与 dock 对齐、货箱接近静止，并连续保持一小段稳定时间**。次目标是**轻柔操作**（避免多次硬碰撞导致货箱损坏）和**不越界**（小车与货箱都不能离开仓库地面）。不该混淆的目标：单纯“碰到 dock 区域”不算成功；单纯“把货箱推得离 dock 更近”也不等于完成，因为货箱会因地面阻尼继续滑行，小车无法从后方减速货箱，最终必须满足低速+对齐+静止的停靠条件。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: high
reason: 核心目标是“把物体（货箱）移动到指定目标位姿（dock 内、朝向对齐、静止）”，属于典型的物体搬运/操控到目标位姿任务。虽然载体是轮式小车而非机械臂，且没有夹爪（只能靠推），但任务本质是 staged manipulation：接近货箱 → 推动货箱穿过开口 → 在 dock 内低速对齐停靠。附属的“轻柔”“不越界”“省时”都是约束/次目标，不构成多目标冲突，因此不选 multi_objective_task。

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
- obs[14]: cart_crate_contact，小车与货箱当前是否接触（1/0），reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=远，1=接触），reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例 [0,1]，reward_usable: true（可用于时间相关 shaping，但需谨慎）

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [2]
- bounds: 每通道 [-1.0, 1.0]
- action[0] drive：沿小车朝向的纵向力指令；+1 前进，-1 倒车
- action[1] steer：转向力矩指令；+1 左转（逆时针），-1 右转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `docked_success` —— 货箱完整位于 dock 内、朝向误差 < 30°、速度 < 0.05 m/s，且连续保持 10 个环境步。
- failure-like termination: `crate_out_of_bounds`（货箱中心离开仓库地面矩形）、`cart_out_of_bounds`（小车中心离开仓库地面矩形）、`crate_damaged`（硬碰撞计数 ≥ 3，硬碰撞定义为峰值法向冲量超过易碎阈值的车-箱接触）。
- ambiguous termination: 无显式歧义终止；但“货箱进入 dock 但未满足对齐/静止/保持”不会终止，属于未完成状态。
- truncation: `time_limit` —— 达到固定步数预算，报告为 truncation，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被禁止使用）
- explicit_failure_flag_available: false（info 中 `termination_reason`、`hard_collision_count` 等被禁止使用）
- allowed_info_fields: []（无任何允许的 info 字段）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

> 注意：成功/失败/损坏/越界等事件**不能**从 info 读取，但可从观测间接推断（见第 7 节 derived_possible）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（19 维观测向量，按第 3 节索引解释）
- action（2 维连续动作）
- next_obs（下一步观测，用于计算 delta / 变化量）
- info 中明确允许的字段：**无**（allowed_info_fields 为空）
- training_progress：仅当 prompt 明确允许时才用；本环境未声明允许，默认不使用

禁止使用：
- original_reward（官方奖励被 mask）
- official_reward / masked_reward
- 未声明的 info 字段（尤其 is_success、termination_reason、hard_collision_count、contact_impulse、cargo_inside_dock、stable_steps 等）
- 未声明的 obs 切片（只能使用 obs[0..18]）

## 7. 可用于奖励函数的信号
- position:
  - 小车世界位置：由 obs[0]*5.0, obs[1]*4.0 恢复（仓库半宽 5.0，半高 4.0）。
  - 货箱世界位置：将 (obs[6]*3.0, obs[7]*3.0) 按小车朝向旋转后加上小车位置。
  - 货箱到 dock 偏移：obs[12]*5.0, obs[13]*4.0（直接可用）。
  - derived_possible：货箱是否“接近 dock 中心”可由 obs[12], obs[13] 的模长推断；货箱是否“完整进入 dock”需结合货箱尺寸与 dock 尺寸（尺寸未在 obs 中显式给出，只能近似用偏移阈值推断）。
- velocity:
  - 小车前向速度 obs[4]*3.0；小车角速度 obs[5]*8.0。
  - 货箱世界速度 obs[8]*3.0, obs[9]*3.0；货箱轴向速度 = sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)。
  - derived_possible：货箱“接近静止”可由货箱速度模长 < 0.05 推断。
- orientation:
  - 小车朝向由 obs[2], obs[3] 给出。
  - 货箱朝向由 obs[10], obs[11] 给出；货箱朝向误差 = atan2(obs[11], obs[10])（相对世界系，需与 dock 朝向比较）。
  - derived_possible：货箱朝向是否对齐 dock（误差 < 30°）可由 obs[10], obs[11] 推断。
- contact:
  - obs[14] 车-箱接触标志（1/0）。
  - obs[15..17] 前/左/右静态障碍接近度（可用于避墙、避免撞隔墙）。
  - derived_possible：硬碰撞/损坏无法直接读取，但可用“接触 + 相对速度/接近度突变”间接近似（不可靠，需谨慎）。
- action/engine:
  - action[0] drive、action[1] steer 可用于动作平滑/能耗 shaping（仅在任务要求高效时）。
- other:
  - obs[18] time_fraction 可用于时间相关 shaping（谨慎，避免鼓励拖延或过早放弃）。

## 8. 不确定或不可用的信号
- 官方奖励 masked_reward：不可用。
- info 全部字段：不可用（is_success、termination_reason、hard_collision_count、contact_impulse、cargo_inside_dock、stable_steps、stagnation_steps、action_energy、component_returns 等）。
- 货箱尺寸、dock 矩形尺寸、隔墙开口位置：obs 未显式给出，只能间接推断。
- 硬碰撞计数与损坏阈值：不可直接读取，只能从接触与速度突变间接近似，可靠性低。
- 精确的“稳定保持步数”：不可直接读取，只能通过连续多步观测自行维护内部计数（若允许在 reward 内维护状态，需谨慎）。
- 隔墙开口的精确几何：不可直接读取，只能通过 sensor_front/left/right 与位置推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation
control_type: continuous
morphology:
  body_type: wheeled_cart_top_down
  actuator_type: longitudinal_force_plus_steering_torque
  contact_structure: non_grasping_push_contact_with_freely_sliding_crate
primary_objectives:
  - 将货箱完整送入 dock 矩形内
  - 使货箱朝向与 dock 对齐（误差 < 30°）
  - 使货箱接近静止（速度 < 0.05 m/s）并保持短稳定期
secondary_objectives:
  - 轻柔操作，避免多次硬碰撞导致货箱损坏
  - 小车与货箱均不越出仓库地面
  - 在时间预算内完成
main_failure_risks:
  - 硬碰撞累计 ≥ 3 导致货箱损坏终止
  - 货箱或小车越界终止
  - 货箱滑过 dock 无法减速（小车无刹车、无法从后方减速货箱）
  - 货箱卡在隔墙开口或撞墙
  - 时间耗尽（truncation，非成功）
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向 dock 中心靠近，提供主进度信号。
  why_required: 任务核心是把货箱送到 dock，必须有指向目标的进度信号。
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13]]
  risks: 若只用 proximity（距离本身）会鼓励悬停在“较近但未完成”状态；应优先用 delta(distance) 或 improvement。
- role_id: crate_dock_alignment
  purpose: 促使货箱朝向与 dock 对齐。
  why_required: 成功条件要求朝向误差 < 30°，仅位置到位不够。
  usable_signals: [obs[10], obs[11], next_obs[10], next_obs[11]]
  risks: 朝向误差需相对 dock 朝向计算；dock 朝向未显式给出，需假设或从几何推断。
- role_id: crate_settling
  purpose: 促使货箱在 dock 内接近静止。
  why_required: 成功条件要求货箱速度 < 0.05 m/s 并保持稳定。
  usable_signals: [obs[8], obs[9], next_obs[8], next_obs[9]]
  risks: 小车无法从后方减速货箱，若奖励要求“一直推到静止”会与物理冲突；应奖励货箱自身低速状态，而非持续推力。

### 10.2 条件职责 conditional_roles
- role_id: gentle_contact
  condition_to_use: 当需要抑制硬碰撞、保护易碎货箱时加入。
  usable_signals: [obs[14], obs[4], obs[8], obs[9], obs[15], obs[16], obs[17]]
  risks: 硬碰撞计数不可直接读取，只能用接触 + 相对速度近似；近似不可靠，可能误罚正常推动。
- role_id: boundary_avoidance
  condition_to_use: 当需要避免小车/货箱越界时加入。
  usable_signals: [obs[0], obs[1], obs[12], obs[13], obs[15], obs[16], obs[17]]
  risks: 越界终止不可直接读取，只能用位置接近边界推断；阈值需谨慎设定。
- role_id: action_smoothness_or_energy
  condition_to_use: 仅当任务明确要求高效/平滑时加入。
  usable_signals: [action[0], action[1]]
  risks: 本任务描述未强调节能，属附属优化，默认不加。
- role_id: time_efficiency
  condition_to_use: 仅当需要鼓励尽快完成时加入。
  usable_signals: [obs[18]]
  risks: 可能鼓励冒险高速操作，增加损坏风险；默认不加。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用。
  forbidden_or_missing_signals: [original_reward, masked_reward, official_reward_terms]
- role_id: info_based_success_bonus
  reason: info 中 is_success、termination_reason、cargo_inside_dock、stable_steps 等被禁止使用。
  forbidden_or_missing_signals: [is_success, termination_reason, cargo_inside_dock, stable_steps]
- role_id: hard_collision_penalty_from_info
  reason: hard_collision_count、contact_impulse 被禁止使用；只能间接近似，可靠性低。
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse]
- role_id: push_until_stop
  reason: 小车无刹车、无法从后方减速货箱，持续推力无法实现低速停靠，与物理冲突。
  forbidden_or_missing_signals: [brake_capability]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 无 | delta(distance), improvement, bounded_signal | 用 delta 避免悬停陷阱；距离可由偏移模长计算 |
| crate_dock_alignment | obs[10], obs[11], next_obs[10], next_obs[11] | dock 朝向（需假设/推断） | bounded_signal, cosine_alignment | 朝向误差 = atan2(obs[11], obs[10]) 与 dock 朝向比较 |
| crate_settling | obs[8], obs[9], next_obs[8], next_obs[9] | 无 | bounded_signal, quadratic_penalty（仅超阈值时） | 奖励货箱低速状态，而非持续推力 |
| gentle_contact | obs[14], obs[4], obs[8], obs[9], obs[15..17] | hard_collision_count, contact_impulse | hinge_penalty, gated_signal | 用接触 + 相对速度近似硬碰撞，可靠性低 |
| boundary_avoidance | obs[0], obs[1], obs[12], obs[13], obs[15..17] | 越界标志 | hinge_penalty, gated_signal | 用位置接近边界推断越界风险 |
| action_smoothness_or_energy | action[0], action[1] | 无 | quadratic_penalty, bounded_signal | 默认不加，仅任务要求时 |
| time_efficiency | obs[18] | 无 | bounded_signal | 默认不加，可能鼓励冒险 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱被推过 dock 无法减速 | 货箱速度在接近 dock 时仍高，随后越过 dock 中心 | 增加货箱低速/接近 dock 时的减速 shaping；奖励在 dock 附近降低推力 |
| 货箱卡在隔墙开口或撞墙 | sensor_front/left/right 持续高值，货箱位置停滞 | 增加避墙 shaping 或引导绕行开口 |
| 硬碰撞累计导致损坏终止 | 接触频繁且相对速度高，episode 提前终止 | 增加轻柔接触 shaping，抑制高速撞击 |
| 小车或货箱越界 | 位置接近边界，episode 提前终止 | 增加边界 hinge penalty |
| 货箱朝向未对齐 | 货箱朝向误差持续 > 30°，无法满足成功条件 | 增加朝向对齐 shaping |
| 悬停在 dock 附近但不完成 | 货箱距离 dock 近但速度/朝向不满足，episode 拖到 truncation | 用 delta 主信号替代 proximity，避免悬停收割 |
| 时间耗尽（truncation） | obs[18] 接近 1，未触发成功终止 | 检查主信号是否过弱或探索不足；必要时加时间效率 shaping |
| 主信号被悬停收割 | 货箱停在较近位置持续获得正分 | 改用 delta(distance) 或 improvement 作为主信号 |