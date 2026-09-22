# 匿名环境理解卡片

## 1. 任务目标
该匿名环境是一个俯视仓储推箱任务：一辆无刹车、无夹爪的轮式小车在平坦地面上，通过纵向驱动力和转向力矩，将一只易碎方形货箱从墙近侧、穿过隔墙上的窄开口，推送到远侧的矩形交付停靠区（dock）。主目标不是小车自身到达某点，而是货箱最终完全进入 dock 矩形、朝向与 dock 对齐、速度接近零，并连续保持 10 个环境步。次目标包括避免货箱受到 3 次以上硬冲击、避免货箱或小车越出仓库边界、以及在不损坏货箱的前提下通过窄开口。时间耗尽只触发截断，不算成功；货箱滑行是地板阻尼所致，小车无法从后方拉拽或刹车货箱。

## 2. 任务类型选择
selected_route_id: manipulation_grasping  
confidence: high  
reason: 核心目标是把一个可自由移动的物体（货箱）操控到指定位置和位姿，并满足低速稳定条件；动作是连续力/力矩，接触推动是唯一移动货箱的方式，属于典型的物体操控到指定位姿。不是 navigation_goal_reaching，因为成功判据在货箱而非小车；也不是 locomotion_continuous_control，因为不是通过地形持续前进；也不是 multi_objective_task，因为安全/不损坏是约束而非与主目标权重相当的核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 所有维度 clipped 到 [-2.0, 2.0]
- obs[0]: cart_x，小车 x 位置除以仓库半宽，0 = 中心线，+1 = 远侧墙方向；reward_usable: true
- obs[1]: cart_y，小车 y 位置除以仓库半高，0 = 中心线；reward_usable: true
- obs[2]: cart_cos_heading，小车航向角余弦；reward_usable: true
- obs[3]: cart_sin_heading，小车航向角正弦；reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身航向速度除以 3.0 m/s；reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度除以 8.0 rad/s；reward_usable: true
- obs[6]: crate_rel_x_body，货箱相对小车在车体坐标系 x 分量除以 3.0 m；reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体坐标系 y 分量除以 3.0 m；reward_usable: true
- obs[8]: crate_vx，货箱世界坐标系线速度 x 除以 3.0 m/s；reward_usable: true
- obs[9]: crate_vy，货箱世界坐标系线速度 y 除以 3.0 m/s；reward_usable: true
- obs[10]: crate_cos_heading，货箱航向角余弦；reward_usable: true
- obs[11]: crate_sin_heading，货箱航向角正弦；reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到 dock 中心的有符号 x 偏移除以仓库半宽（5.0 m）；reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到 dock 中心的有符号 y 偏移除以仓库半高（4.0 m）；reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱当前是否接触，1.0 = 接触，0.0 = 未接触；reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度，0 = 范围内无遮挡，1 = 触碰；reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，0 = 范围内无遮挡，1 = 触碰；reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，0 = 范围内无遮挡，1 = 触碰；reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例，[0, 1]；reward_usable: true（但只应用于条件判断/截断感知，不宜直接作为正奖励来源）

补充几何关系：
- 货箱世界坐标可由 cart_x、cart_y、cart 航向、obs[6]、obs[7] 恢复。
- crate_to_dock 实际米制偏移：x 米 = obs[12] * 5.0，y 米 = obs[13] * 4.0。
- “完全在 dock 内”的阈值：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030，对应实际 |x| <= 0.12 m 且 |y| <= 0.12 m。
- 货箱轴向速度 = sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)。
- 货箱航向角 = atan2(obs[11], obs[10])。

## 4. 动作空间 action_space
- type: Box
- shape: [2]
- continuous: true
- bounds: 每个通道 [-1.0, 1.0]
- action[0]: drive，沿小车航向的纵向力命令；+1 向前驱动，-1 反向驱动
- action[1]: steer，转向力矩命令；+1 左转（逆时针），-1 右转
- 物理含义：小车每步接收纵向力与转向力矩，无独立刹车；货箱只能通过接触被推动。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - docked_success：货箱完全在 dock 内、航向误差小于 30 度、速度低于 0.05 m/s，并连续保持 10 个环境步。该条件下 terminated=True。
- failure-like termination:
  - crate_out_of_bounds：货箱中心离开仓库地面矩形。
  - cart_out_of_bounds：小车中心离开仓库地面矩形。
  - crate_damaged：货箱累计受到 3 次或以上硬冲击；硬冲击定义为小车-货箱接触的峰值法向冲量超过易碎阈值。
- ambiguous termination:
  - time_limit：达到固定步数预算时 truncated=True，不是任务成功。
- truncation:
  - time_limit 明确报告为截断，不应被当作成功。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中即使存在 is_success，也属于禁止字段）
- explicit_failure_flag_available: false（info 中 termination_reason、hard_collision_count 等被禁止）
- allowed_info_fields: []（没有允许使用的 info 字段）
- forbidden_or_uncertain_info_fields:
  - is_success
  - cargo_goal_distance
  - cargo_angle_error
  - cargo_speed
  - robot_cargo_distance
  - contact_impulse
  - hard_collision_count
  - stagnation_steps
  - action_energy
  - component_returns
  - official_reward_terms
  - termination_reason
  - cargo_inside_dock
  - stable_steps
- 间接推断路径（derived_possible）：
  - success-like：obs[12]、obs[13] 的绝对值同时满足阈值，货箱速度接近零，航向误差小，且 episode 终止但未观察到出界或反复硬接触。
  - crate_out_of_bounds：货箱世界坐标超出仓库合理范围；obs[12]、obs[13] 或恢复后的货箱世界坐标异常。
  - cart_out_of_bounds：obs[0]、obs[1] 超出合理范围。
  - crate_damaged：无法直接读取 hard_collision_count，但可通过 obs[14] 接触信号与 cart/crate 速度突变组合间接推断；若 episode 在未出界情况下提前终止且此前出现高相对速度接触，可能对应损坏终止。
  - time_limit：obs[18] 接近 1.0 且 episode 以 truncated 结束。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：完整 19 维观测，已声明维度含义。
- action：2 维连续动作。
- next_obs：下一步 19 维观测。
- info 中明确允许的字段：本环境 allowed_info_fields 为空，因此不可以使用任何 info 字段。
- training_progress：本 prompt 未明确允许使用，因此禁止使用。

禁止使用：
- original_reward
- official_reward
- 任何未声明的 info 字段
- 任何未声明的 obs 切片
- info 中的 is_success、cargo_goal_distance、cargo_angle_error、cargo_speed、robot_cargo_distance、contact_impulse、hard_collision_count、stagnation_steps、action_energy、component_returns、official_reward_terms、termination_reason、cargo_inside_dock、stable_steps

## 7. 可用于奖励函数的信号
- position:
  - cart_x: obs[0]
  - cart_y: obs[1]
  - crate_rel_x_body: obs[6]
  - crate_rel_y_body: obs[7]
  - crate_to_dock_x: obs[12]
  - crate_to_dock_y: obs[13]
  - 可由 obs[0]、obs[1]、obs[2]、obs[3]、obs[6]、obs[7] 恢复货箱世界坐标。
- velocity:
  - cart_forward_speed: obs[4]
  - cart_yaw_rate: obs[5]
  - crate_vx: obs[8]
  - crate_vy: obs[9]
  - 货箱速度大小可由 obs[8]、obs[9] 计算。
- orientation:
  - cart_cos_heading: obs[2]
  - cart_sin_heading: obs[3]
  - crate_cos_heading: obs[10]
  - crate_sin_heading: obs[11]
  - 货箱航向角 = atan2(obs[11], obs[10])；可用于推断与 dock 轴的对齐程度。
- contact:
  - cart_crate_contact: obs[14]
  - sensor_front: obs[15]
  - sensor_left: obs[16]
  - sensor_right: obs[17]
  - 接触时结合 cart/crate 速度可间接推断相对冲击强度，但无法直接获得冲量。
- action/engine:
  - action[0]: drive
  - action[1]: steer
  - 可计算动作幅度、平滑度、能耗代理。
- other:
  - time_fraction: obs[18]
  - 可用于截断感知，但不应直接作为正奖励来源；任务未要求时间效率。

## 8. 不确定或不可用的信号
- info 中所有字段均不可用，包括但不限于：is_success、cargo_goal_distance、cargo_angle_error、cargo_speed、robot_cargo_distance、contact_impulse、hard_collision_count、stagnation_steps、action_energy、component_returns、official_reward_terms、termination_reason、cargo_inside_dock、stable_steps。
- original_reward / masked_reward 不可用。
- 官方 reward 组件不可用、不可回忆、不可复现。
- 直接接触冲量 contact_impulse 不可用；只能从 obs[14] 接触状态与 cart/crate 速度变化间接推断相对冲击风险。
- 硬碰撞次数 hard_collision_count 不可用；只能从接触时高相对速度、速度突变、episode 提前终止等组合间接推断。
- termination_reason 不可用；只能通过观测推断成功/失败/截断。
- is_success 不可用。
- cargo_inside_dock 不可用；但可由 obs[12]、obs[13] 的阈值条件间接推断。
- stable_steps 不可用；只能通过连续多步观测自行维护内部计数。
- stagnation_steps 不可用。
- action_energy 不可用。
- dock 的目标朝向未直接暴露；只能从 dock 矩形轴与世界轴对齐这一几何事实推断货箱应对齐到轴方向，可能允许 90 度倍数。
- 隔墙开口的精确几何未直接暴露；只能从传感器 obs[15]-obs[17] 和 cart/crate 位置变化间接推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation
control_type: continuous
morphology:
  body_type: wheeled cart
  actuator_type: longitudinal force and steering torque
  contact_structure: pushing contact with freely-moving fragile crate; static partition wall with narrow opening
primary_objectives:
  - 将货箱推入 dock 矩形并完全包含
  - 使货箱朝向与 dock 对齐
  - 使货箱速度低于 0.05 m/s 并连续稳定 10 个环境步
secondary_objectives:
  - 避免货箱受到 3 次以上硬冲击
  - 避免货箱或小车越出仓库边界
  - 通过隔墙窄开口
main_failure_risks:
  - 硬冲击导致货箱损坏并提前终止
  - 货箱滑行失控出界
  - 小车或货箱越界
  - 在 dock 内未对齐或速度过快导致无法稳定
  - 时间耗尽截断
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 推动货箱向 dock 中心接近，提供可学习的前进梯度。
  why_required: 主目标是货箱到达 dock；若没有接近信号，稀疏成功事件难以探索。
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13]]
  risks: 单独使用 proximity 可能让智能体停在中间状态悬停收割；应优先使用两步 delta / improvement，并配合 dock 内条件或安全 gate。

- role_id: dock_containment
  purpose: 奖励货箱完全进入 dock 矩形，而不仅仅是中心接近。
  why_required: 成功条件要求货箱完全包含在 dock 内，阈值严格：|obs[12]| <= 0.024 且 |obs[13]| <= 0.030。
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13]]
  risks: 阈值严格，信号稀疏；过早给密集奖励可能导致在阈值外徘徊；应使用 bounded_signal 或条件性 sparse bonus。

- role_id: heading_alignment
  purpose: 促使货箱朝向与 dock 对齐。
  why_required: 成功条件要求航向误差小于 30 度并保持稳定。
  usable_signals: [obs[10], obs[11], next_obs[10], next_obs[11]]
  risks: dock 目标朝向未直接暴露；需推断为与仓库轴对齐，可能允许 90 度倍数；若目标朝向判断错误，会奖励错误姿态。

- role_id: low_speed_settling
  purpose: 促使货箱在 dock 内低速并最终近静止。
  why_required: 成功条件要求货箱速度低于 0.05 m/s 并连续保持 10 步。
  usable_signals: [obs[8], obs[9], next_obs[8], next_obs[9]]
  risks: 过早惩罚速度会阻碍推动阶段；应在接近 dock 或进入 dock 后条件激活，使用 hinge 或 bounded_signal。

- role_id: safety_gate_and_boundary_avoidance
  purpose: 保护主信号，避免硬冲击、越界和货箱损坏。
  why_required: 3 次硬冲击即失败，越界即失败；安全是必要约束。
  usable_signals: [obs[14], obs[15], obs[16], obs[17], obs[0], obs[1], obs[12], obs[13], obs[4], obs[5], obs[8], obs[9]]
  risks: 硬碰撞计数不可直接读，只能间接推断；gate 设计过强可能抑制必要推动动作。

### 10.2 条件职责 conditional_roles
- role_id: wall_proximity_avoidance
  condition_to_use: 当 sensor_front/left/right 接近 1，或小车/货箱靠近隔墙与开口时启用。
  usable_signals: [obs[15], obs[16], obs[17], obs[0], obs[1], obs[6], obs[7]]
  risks: 传感器只反映静态障碍接近度，不区分开口与墙；过强惩罚可能导致不敢通过开口。

- role_id: gentle_pushing
  condition_to_use: 当 obs[14] 接触为 1，且相对速度较高时启用。
  usable_signals: [obs[14], obs[4], obs[5], obs[8], obs[9]]
  risks: 无法直接读接触冲量；用相对速度代理可能不完全等价于硬冲击阈值；过度惩罚会抑制推动。

- role_id: action_smoothness
  condition_to_use: 若训练后发现动作高频抖动、推动不稳定时启用。
  usable_signals: [action[0], action[1]]
  risks: 任务未明确要求动作平滑；过早引入可能限制必要的大推力。

- role_id: time_awareness
  condition_to_use: 仅在需要避免时间耗尽时作为温和截断感知；不建议作为主要奖励。
  usable_signals: [obs[18]]
  risks: 任务未要求省时；把 time_fraction 变成正奖励会导致拖延或提前结束行为。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: brake_crate_from_behind
  reason: 小车无刹车、无夹爪，不能在货箱后方拉拽或主动减速货箱；要求小车在货箱后方刹车会违背物理。
  forbidden_or_missing_signals: [无直接刹车信号；货箱仅受地板阻尼减速]

- role_id: direct_hard_collision_penalty
  reason: info 中 contact_impulse 和 hard_collision_count 被禁止，无法直接读取硬冲击次数或冲量。
  forbidden_or_missing_signals: [contact_impulse, hard_collision_count]

- role_id: success_flag_bonus
  reason: info 中 is_success、cargo_inside_dock、stable_steps 等被禁止，不能使用显式成功标志。
  forbidden_or_missing_signals: [is_success, cargo_inside_dock, stable_steps]

- role_id: official_reward_shaping
  reason: original_reward / official_reward / official_reward_terms 被禁止，不能回忆或复现官方奖励。
  forbidden_or_missing_signals: [original_reward, official_reward_terms]

- role_id: terminal_reason_penalty
  reason: info 中 termination_reason 被禁止，不能直接按终止原因给奖励或惩罚。
  forbidden_or_missing_signals: [termination_reason]

- role_id: proximity_only_hover_reward
  reason: 仅奖励货箱接近 dock 中心会允许智能体停在较好但不满足完全包含和低速稳定的中间状态，形成悬停陷阱。
  forbidden_or_missing_signals: [无缺失信号，但数学形式风险高]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 无直接距离但可计算 | delta(distance), improvement, dense_state_signal | 优先两步 delta，避免 proximity 悬停；可结合安全 gate |
| dock_containment | obs[12], obs[13], next_obs[12], next_obs[13] | cargo_inside_dock | bounded_signal, indicator, hinge | 阈值：|obs[12]|<=0.024 且 |obs[13]|<=0.030；稀疏但准确 |
| heading_alignment | obs[10], obs[11], next_obs[10], next_obs[11] | cargo_angle_error, dock 目标朝向 | cos_similarity, angular_penalty, bounded_signal | 目标朝向未显式暴露；推断为与仓库轴对齐，可能允许 90 度倍数 |
| low_speed_settling | obs[8], obs[9], next_obs[8], next_obs[9] | cargo_speed | bounded_signal, hinge_penalty, quadratic_penalty | 应在接近 dock 或进入 dock 后条件激活；避免阻碍推动阶段 |
| safety_gate_and_boundary_avoidance | obs[14], obs[15], obs[16], obs[17], obs[0], obs[1], obs[12], obs[13], obs[4], obs[5], obs[8], obs[9] | contact_impulse, hard_collision_count, termination_reason | gate, hinge_penalty, bounded_signal | 硬冲击只能间接推断；gate 用于保护主信号 |
| wall_proximity_avoidance | obs[15], obs[16], obs[17], obs[0], obs[1], obs[6], obs[7] | 墙体精确几何 | hinge_penalty, gate | 传感器接近 1 时条件启用；避免抑制通过开口 |
| gentle_pushing | obs[14], obs[4], obs[5], obs[8], obs[9] | contact_impulse | hinge_penalty, bounded_signal | 接触时用相对速度代理；不可直接读冲量 |
| action_smoothness | action[0], action[1] | 无 | quadratic_penalty, delta(action) | 仅在动作抖动导致失败时条件启用 |
| time_awareness | obs[18] | 无 | bounded_signal, gate | 任务未要求省时；不建议作为正奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱接近 dock 但未完全进入 | obs[12]/obs[13] 绝对值长期略大于阈值，episode 未成功终止；crate_to_dock_progress 有信号但 dock_containment 稀疏 | 加强 dock_containment 的条件性稀疏奖励；使用 delta 引导进入阈值；避免 proximity 悬停 |
| 高速冲入 dock 后滑出 | 货箱速度 obs[8]/obs[9] 在 dock 附近仍高；obs[12]/obs[13] 在阈值内停留不足 10 步 | 在接近 dock 时逐步激活 low_speed_settling；用 hinge 惩罚高速进入；鼓励提前释放推动 |
| 因硬冲击提前终止 | episode 早期终止且未出界；obs[14] 接触时 cart/crate 相对速度高；速度出现突变 | 加入 gentle_pushing 条件职责；接触时惩罚高相对速度；用 safety gate 抑制高速接触 |
| 卡在墙边或开口处 | sensor_front/left/right 长期接近 1；cart 位置停滞；crate_to_dock_progress 接近零 | 加入 wall_proximity_avoidance 条件职责；调整 gate 强度；鼓励通过开口时低速微调 |
| 货箱被推出界 | 恢复的货箱世界坐标超出合理范围；obs[12]/obs[13] 剧烈偏离；episode 提前终止 | 强化 boundary gate；使用 hinge 惩罚接近边界；避免过大推力 |
| 小车出界 | obs[0]/obs[1] 超出合理范围；episode 提前终止 | 加入小车位置边界 gate；动作幅度约束 |
| 时间耗尽未完成 | obs[18] 接近 1.0；episode 以截断结束；crate_to_dock_progress 不足 | 检查主信号是否太稀疏；适当增加接近阶段引导；不建议直接奖励剩余时间 |
| 货箱朝向不对齐 | obs[10]/obs[11] 计算的货箱航向长期偏离轴方向；成功终止未出现 | 加入 heading_alignment 条件职责；确认目标朝向是否允许 90 度倍数；用角度 hinge 惩罚 |
| 策略不敢推货箱 | obs[14] 长期为 0；货箱位置几乎不变；crate_to_dock_progress 无进展 | 检查 safety gate 是否过强；降低接触惩罚；增加接近阶段的 dense 引导 |
| 动作高频抖动导致推动不稳定 | action[0]/action[1] 高频大幅变化；货箱速度波动大 | 条件性加入 action_smoothness；降低动作变化惩罚权重，避免抑制必要推力 |