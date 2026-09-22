# Prompt Record

## System Prompt

```text
你是奖励函数设计专家。上一版奖励函数已经完成了若干训练校验，下面给出它的**训练反馈（reward reflection）**。请据此改进这个奖励函数。

# 你的任务

写出**改进后的完整奖励函数**。可以修改权重、阈值、数学形态，可以新增、删除或合并组件。多改少改由你判断，但必须在代码里能被验证。

# 决策依据

1. **任务分数**：`mean_eval_reward` 是用环境原生目标衡量的策略表现。它上升说明方向对，停滞或下降说明当前奖励在诱导错误行为。
2. **组件数值**：下表是各组件在训练过程中的取值。请判断：
   - 某个组件始终为 0 或几乎不触发 → 它可能永远无法被满足，或者设定得太稀疏；
   - 某个组件数值极大、压过其余组件 → 它可能在主导策略行为；
   - 总奖励的走向与任务分数不一致 → 代理指标与真正目标脱节，需要重新对齐。
3. **不要只看总奖励**。总奖励是策略在优化你自己写的目标；任务分数才代表它有没有真的完成任务。

# 输出要求

先写一段简短分析（不超过 5 句，说明你看出的问题和打算怎么改），然后立即输出完整 Python 代码。

# 必须避免的激励冲突（违反即无效）

任何辅助组件（低速、稳定、对齐、平滑等）都**不允许因为"推进任务所必需的动作"而下降**。

典型反例：用 `k / (1 + c * speed)` 形式的"低速因子"作为**持续的全局状态奖励**。推动目标物体必然产生速度，于是该组件在惩罚任务进展本身；一旦它的量级压过主信号，最优策略就变成"什么都不做"。

要求：
- "低速 / 静止 / 对齐"这类量只能作为**门控**，**不得作为全局持续奖励**被动收分；
- 任何组件都不得对"把目标推向目标位置"这一行为产生净负贡献。

**必做自检**：把奖励分别代入 ①什么都不做、目标静止在初始位置；②正在把目标推向目标位置（目标因此具有速度）。要求 **②的单步总奖励严格高于 ①**。

# 代码输出格式

只输出 Python 函数本体，不要解释、不要 markdown 代码块、不要注释以外的任何文字。

# 代码约束

- 只用环境事实声明的观测维度与索引。
- 禁止 `terminal_success_reward`、`terminal_failure_penalty`、`original_reward`。
- 禁止 `import`、`class`、`try/except`、`lambda`、`eval/exec/open`。
- 平方根 `** 0.5`；指数 `2.718281828 ** exponent`。
- 函数签名必须逐字符保持：
  `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 必须给 `components` 字典赋值，并返回 `(float(total_reward), components)`。

```

## User Prompt

```markdown
# Environment card

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
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_FAILED_PAID = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 自检记录（估算，基于典型每步距离变化）：
    # R_idle ≈ -0.002（仅时间成本）
    # R_push ≈ +0.35（推进项约 0.4，动作/时间/轻柔代理约 -0.05）
    # R_settled ≈ +1.00（停稳每步 +1.0，时间/动作约 -0.004）
    # 满足 R_push > R_idle，R_settled > R_push。
    # 自检③：closing=1.0 时 roughness≈-0.50，closing=0.05 时≈-0.025，差距 0.475，
    # 与正常推进项（约 0.4/步）同量级。

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED_PAID[0] = False
    _PREV_T[0] = t

    # 小车→货箱距离（米）
    prev_cart_crate = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    next_cart_crate = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = 1.0 * (prev_cart_crate - next_cart_crate)

    # 货箱→泊位距离（米）
    prev_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    next_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = 1.0 * (prev_dock - next_dock)

    # 首次完全进入泊位容差
    dock_enter = 0.0
    if (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030 and not _ENTERED[0]):
        _ENTERED[0] = True
        dock_enter = 5.0

    # 接触轻柔度代理（roughness）
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.5 * contact * closing

    # 硬冲击固定惩罚与计数
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 动作代价与时间成本
    action_cost = -0.0005 * (action[0] ** 2 + action[1] ** 2)
    time_cost = -0.002

    # 完成谓词：泊位内 + 对齐 + 慢
    speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align = max(abs(next_obs[10]), abs(next_obs[11]))
    in_dock = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030)
    done_now = in_dock and (align >= 0.8660254) and (speed < 0.05)

    # 停稳期每步收益（不因一次性事件或连续计数而关闭）
    settle_bonus = 1.0 if done_now else 0.0

    # 连续 10 步完成计数
    if done_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 边界守卫（小车位置归一化，超过 0.95 开始惩罚）
    pen_x = max(0.0, abs(next_obs[0]) - 0.95)
    pen_y = max(0.0, abs(next_obs[1]) - 0.95)
    boundary_guard = -50.0 * (pen_x + pen_y)

    # 失败一次性：越界或累计硬冲击 >= 3
    terminal_failure = 0.0
    if not _FAILED_PAID[0]:
        cart_out = (abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05)

        cart_x = next_obs[0] * 5.0
        cart_y = next_obs[1] * 4.0
        cos_h = next_obs[2]
        sin_h = next_obs[3]
        rel_x = next_obs[6] * 3.0
        rel_y = next_obs[7] * 3.0
        crate_x = cart_x + rel_x * cos_h - rel_y * sin_h
        crate_y = cart_y + rel_x * sin_h + rel_y * cos_h
        crate_out = (abs(crate_x) > 5.25 or abs(crate_y) > 4.2)

        if cart_out or crate_out or _HARD_HITS[0] >= 3:
            _FAILED_PAID[0] = True
            terminal_failure = -100.0

    total = (approach_cargo + progress + dock_enter + roughness + action_cost +
             time_cost + hard_hit + terminal_success + terminal_failure +
             boundary_guard + settle_bonus)

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
        "boundary_guard": boundary_guard,
        "settle_bonus": settle_bonus,
    }
    return float(total), components
```

# Reward reflection of the current reward (native task score = 23.6572)
### Task score

- mean_eval_reward: 23.657193536123362
- mean_episode_length: 391.15
- eval episodes: 20
- termination breakdown: {'terminated': 1, 'truncated': 19}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 17.00 | 348.8 |
| 33% | 17.58 | 347.0 |
| 50% | 17.56 | 347.0 |
| 67% | 16.97 | 349.8 |
| 83% | 17.70 | 346.6 |
| 100% | 17.61 | 346.6 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminal_success | 0.000 | 0.399 | 0.800 | 1.600 | 0.800 | 121.247 | 237.912 | 213.035 | 184.898 | 158.805 | 131.250 |
| total_reward | -2.343 | -0.794 | -0.070 | 1.270 | 3.441 | 18.862 | 31.750 | 32.334 | 34.090 | 33.868 | 33.951 |
| settle_bonus | 0.000 | 0.013 | 0.036 | 0.053 | 0.029 | 4.212 | 8.801 | 10.539 | 14.058 | 15.755 | 17.312 |
| dock_enter | 0.000 | 0.146 | 0.027 | 0.420 | 2.227 | 4.105 | 4.497 | 4.703 | 4.730 | 4.594 | 4.844 |
| progress | 0.220 | 0.604 | 1.307 | 1.557 | 2.709 | 3.754 | 3.897 | 3.903 | 3.857 | 3.848 | 3.858 |
| approach_cargo | 0.415 | 0.907 | 0.983 | 0.983 | 1.050 | 1.134 | 1.145 | 1.200 | 1.197 | 1.192 | 1.097 |
| time_cost | -0.799 | -0.798 | -0.799 | -0.799 | -0.799 | -0.694 | -0.564 | -0.584 | -0.612 | -0.639 | -0.676 |
| roughness | -0.627 | -0.958 | -0.770 | -0.834 | -1.454 | -1.147 | -0.903 | -0.764 | -0.689 | -0.725 | -0.627 |
| action_cost | -0.201 | -0.215 | -0.213 | -0.212 | -0.214 | -0.182 | -0.151 | -0.156 | -0.160 | -0.166 | -0.170 |
| boundary_guard | -1.207 | -0.334 | -0.617 | 0.000 | -0.030 | 0.000 | -0.027 | 0.000 | 0.000 | -0.032 | 0.000 |
| terminal_failure | -0.933 | -1.064 | -0.400 | 0.000 | -0.667 | 0.000 | -0.094 | 0.000 | 0.000 | -0.107 | 0.000 |
| hard_hit | -0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminal_success | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.3% | 0.2% | 0.2% | 0.2% | 0.1% |
| total_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settle_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.2% | 3.1% | 3.6% | 4.6% | 4.9% | 5.1% |
| dock_enter | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.2% | 0.3% | 0.3% | 0.3% | 0.3% | 0.3% |
| progress | 22.2% | 43.8% | 43.7% | 44.0% | 58.6% | 60.1% | 63.4% | 68.7% | 73.1% | 73.5% | 74.1% |
| approach_cargo | 100.0% | 99.9% | 99.9% | 99.8% | 99.7% | 99.7% | 99.6% | 99.6% | 99.5% | 99.4% | 99.6% |
| time_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| roughness | 1.8% | 5.1% | 5.8% | 8.9% | 17.8% | 18.0% | 19.8% | 19.1% | 17.4% | 16.3% | 14.0% |
| action_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| boundary_guard | 0.2% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| hard_hit | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| action_cost | -0.1836 | 0.1836 | -0.3413 | -0.1062 |
| approach_cargo | 1.0411 | 1.0927 | -3.5144 | 2.4354 |
| boundary_guard | -0.1963 | 0.1963 | -414.0414 | 0.0000 |
| dock_enter | 2.8159 | 2.8159 | 0.0000 | 5.0000 |
| hard_hit | -0.0005 | 0.0005 | -1.0000 | 0.0000 |
| progress | 2.7390 | 2.7499 | -3.1166 | 4.6078 |
| roughness | -0.8776 | 0.8776 | -9.4458 | 0.0000 |
| settle_bonus | 6.1249 | 6.1249 | 0.0000 | 47.0000 |
| terminal_failure | -0.2888 | 0.2888 | -100.0000 | 0.0000 |
| terminal_success | 105.5440 | 105.5440 | 0.0000 | 300.0000 |
| time_cost | -0.6952 | 0.6952 | -0.8000 | -0.3940 |
| total_reward | 17.4035 | 18.2863 | -415.8834 | 75.1949 |

### IMPORTANT: your previous draft failed validation
- 没有发现 components/reward_components/reward_terms 字典赋值
```
