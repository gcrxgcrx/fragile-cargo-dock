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
    # ---------- helpers ----------
    def _sqrt(x):
        return x ** 0.5

    def _atan2(y, x):
        # small-angle-free atan2 via math-free approximation is not needed;
        # use the built-in through arithmetic identity not available -> use math
        # Since imports are forbidden, we implement atan2 via a series-free method:
        # However Python builtins do not include atan2. Use a rational approximation.
        # Use a robust polynomial approximation of atan on [-1,1] then quadrant fix.
        if x == 0.0 and y == 0.0:
            return 0.0
        ax = x if x >= 0.0 else -x
        ay = y if y >= 0.0 else -y
        if ax >= ay:
            t = y / x if x != 0.0 else 0.0
            # atan(t) for |t|<=1 by series (converges reasonably for |t|<=1)
            t2 = t * t
            at = t * (1.0 - t2 / 3.0 + t2 * t2 / 5.0 - t2 * t2 * t2 / 7.0
                      + t2 * t2 * t2 * t2 / 9.0 - t2 * t2 * t2 * t2 * t2 / 11.0)
            ang = at if x > 0.0 else (at + 3.141592653589793 if y >= 0.0 else at - 3.141592653589793)
        else:
            t = x / y if y != 0.0 else 0.0
            t2 = t * t
            at = t * (1.0 - t2 / 3.0 + t2 * t2 / 5.0 - t2 * t2 * t2 / 7.0
                      + t2 * t2 * t2 * t2 / 9.0 - t2 * t2 * t2 * t2 * t2 / 11.0)
            base = 1.5707963267948966 - at
            ang = base if y > 0.0 else base - 3.141592653589793
        return ang

    def _cos(a):
        # cos via Taylor around 0 after range reduction
        two_pi = 6.283185307179586
        x = a - two_pi * int(a / two_pi)
        if x > 3.141592653589793:
            x = x - two_pi
        if x < -3.141592653589793:
            x = x + two_pi
        x2 = x * x
        return 1.0 - x2 / 2.0 + x2 * x2 / 24.0 - x2 * x2 * x2 / 720.0 + x2 * x2 * x2 * x2 / 40320.0

    def _clamp(v, lo, hi):
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    # ---------- geometry ----------
    # crate offset to dock (world-aligned, normalized by half extents)
    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    ndx = next_obs[12] * 5.0
    ndy = next_obs[13] * 4.0

    dist = _sqrt(dx * dx + dy * dy)
    ndist = _sqrt(ndx * ndx + ndy * ndy)

    # crate speed (world)
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = _sqrt(cvx * cvx + cvy * cvy)

    ncvx = next_obs[8] * 3.0
    ncvy = next_obs[9] * 3.0
    ncrate_speed = _sqrt(ncvx * ncvx + ncvy * ncvy)

    # crate heading error relative to world x-axis (dock assumed axis-aligned)
    crate_ang = _atan2(obs[10], obs[11])  # note: obs[10]=cos, obs[11]=sin -> atan2(sin,cos)
    crate_ang = _atan2(obs[11], obs[10])
    # alignment factor: cos of angle error vs 0 (aligned with world x)
    align_cos = _cos(crate_ang)
    # fold to nearest axis (crate square: 90 deg symmetry) -> use |cos(2*ang)| style
    # simpler: alignment = |cos(ang)| (within 90 deg of axis)
    align = align_cos if align_cos >= 0.0 else -align_cos
    # smooth alignment factor in [0,1]
    align_factor = align

    # ---------- components ----------

    # 1) main progress: reduction of crate-to-dock distance
    progress = _clamp(dist - ndist, -1.0, 1.0)
    c_progress = 6.0 * progress

    # 2) settling: reward low crate speed, but ONLY when crate is near dock
    #    (gated so it never penalizes pushing the crate toward the dock)
    near_gate = _clamp(1.0 - dist / 2.5, 0.0, 1.0)
    speed_factor = 1.0 / (1.0 + 6.0 * crate_speed)
    c_settle = 1.2 * near_gate * speed_factor

    # 3) alignment: reward crate orientation aligned with dock axis, gated by nearness
    c_align = 0.8 * near_gate * align_factor

    # 4) joint completion proxy: near + slow + aligned (geometric mean of factors)
    f_near = _clamp(1.0 - dist / 1.5, 0.0, 1.0)
    f_slow = 1.0 / (1.0 + 20.0 * crate_speed)
    f_align = align_factor
    joint = (f_near * f_slow * f_align) ** (1.0 / 3.0)
    c_joint = 2.5 * joint

    # 5) gentle contact: penalize high relative speed while in contact
    contact = obs[14]
    cart_fwd = obs[4] * 3.0
    rel_speed = _sqrt((cart_fwd - cvx) * (cart_fwd - cvx) + cvy * cvy)
    hard_risk = _clamp(rel_speed - 0.6, 0.0, 3.0)
    c_gentle = -1.5 * contact * hard_risk

    # 6) boundary avoidance: hinge penalty when crate or cart near warehouse edge
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    # crate world position
    cc = obs[2]
    ss = obs[3]
    relx = obs[6] * 3.0
    rely = obs[7] * 3.0
    crate_x = cart_x + cc * relx - ss * rely
    crate_y = cart_y + ss * relx + cc * rely
    crate_margin = 0.6
    cart_margin = 0.6
    c_bound = 0.0
    c_bound -= 3.0 * _clamp(crate_margin - (5.0 - abs(crate_x)), 0.0, 5.0)
    c_bound -= 3.0 * _clamp(crate_margin - (4.0 - abs(crate_y)), 0.0, 4.0)
    c_bound -= 2.0 * _clamp(cart_margin - (5.0 - abs(cart_x)), 0.0, 5.0)
    c_bound -= 2.0 * _clamp(cart_margin - (4.0 - abs(cart_y)), 0.0, 4.0)

    # 7) wall proximity (static obstacles) light penalty
    wall_near = obs[15] if obs[15] > obs[16] else obs[16]
    if obs[17] > wall_near:
        wall_near = obs[17]
    c_wall = -0.4 * _clamp(wall_near - 0.85, 0.0, 1.0)

    components = {
        "crate_to_dock_progress": c_progress,
        "crate_settling": c_settle,
        "crate_dock_alignment": c_align,
        "joint_completion_proxy": c_joint,
        "gentle_contact": c_gentle,
        "boundary_avoidance": c_bound,
        "wall_proximity": c_wall,
    }

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components
```

# Reward reflection of the current reward (native task score = 3.0174)
### Task score

- mean_eval_reward: 3.017388653484644
- mean_episode_length: 400
- eval episodes: 20
- termination breakdown: {'terminated': 0, 'truncated': 20}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 574.91 | 395.1 |
| 33% | 590.69 | 395.8 |
| 50% | 582.39 | 395.2 |
| 67% | 576.27 | 394.3 |
| 83% | 588.69 | 395.8 |
| 100% | 596.81 | 395.4 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 3.392 | 38.398 | 294.015 | 603.456 | 714.494 | 769.348 | 823.149 | 814.713 | 834.104 | 930.536 | 1060.604 |
| joint_completion_proxy | 2.424 | 19.941 | 156.816 | 330.262 | 392.029 | 426.195 | 453.630 | 448.978 | 458.933 | 511.292 | 579.678 |
| crate_settling | 1.925 | 10.994 | 73.804 | 143.762 | 165.743 | 178.089 | 195.328 | 192.056 | 196.684 | 220.858 | 256.307 |
| crate_dock_alignment | 2.344 | 10.022 | 56.238 | 116.933 | 140.752 | 148.511 | 157.139 | 157.093 | 160.399 | 178.715 | 202.764 |
| crate_to_dock_progress | 1.783 | 3.924 | 11.371 | 17.620 | 19.190 | 19.376 | 20.056 | 20.049 | 20.411 | 21.474 | 22.905 |
| gentle_contact | -0.744 | -0.900 | -1.064 | -1.638 | -1.725 | -2.013 | -1.676 | -1.485 | -1.131 | -0.953 | -1.050 |
| boundary_avoidance | -4.340 | -5.583 | -3.149 | -3.482 | -1.493 | -0.808 | -1.327 | -1.978 | -1.191 | -0.850 | 0.000 |
| wall_proximity | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | -0.001 | -0.000 | -0.000 | -0.000 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 20.1% | 32.5% | 63.7% | 83.0% | 85.3% | 87.7% | 85.7% | 84.2% | 84.5% | 85.7% | 88.9% |
| joint_completion_proxy | 1.1% | 5.2% | 27.1% | 53.6% | 63.4% | 67.0% | 65.7% | 63.9% | 65.1% | 68.3% | 73.8% |
| crate_settling | 3.3% | 11.1% | 37.7% | 63.1% | 71.7% | 74.0% | 72.8% | 69.7% | 71.5% | 73.9% | 78.8% |
| crate_dock_alignment | 3.3% | 11.1% | 37.7% | 63.1% | 71.7% | 74.0% | 72.8% | 69.7% | 71.5% | 73.9% | 78.8% |
| crate_to_dock_progress | 18.6% | 28.1% | 45.8% | 51.7% | 49.2% | 48.5% | 45.1% | 46.0% | 45.1% | 43.5% | 42.4% |
| gentle_contact | 0.7% | 0.8% | 1.2% | 2.1% | 2.3% | 2.4% | 2.2% | 1.9% | 1.6% | 1.4% | 1.5% |
| boundary_avoidance | 1.2% | 1.4% | 0.9% | 1.1% | 0.7% | 0.2% | 0.4% | 0.6% | 0.4% | 0.3% | 0.0% |
| wall_proximity | 0.1% | 0.1% | 0.1% | 0.1% | 0.0% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| boundary_avoidance | -2.4093 | 2.4093 | -410.2658 | 0.0000 |
| crate_dock_alignment | 113.2672 | 113.2672 | 0.0000 | 233.8044 |
| crate_settling | 138.5091 | 138.5091 | 0.0000 | 308.3045 |
| crate_to_dock_progress | 15.5684 | 15.6098 | -14.2866 | 27.2308 |
| gentle_contact | -1.3327 | 1.3327 | -81.5793 | 0.0000 |
| joint_completion_proxy | 321.3625 | 321.3625 | 0.0000 | 682.7455 |
| total_reward | 584.9646 | 587.5226 | -279.6786 | 1249.9136 |
| wall_proximity | -0.0007 | 0.0007 | -0.0686 | 0.0000 |
```
