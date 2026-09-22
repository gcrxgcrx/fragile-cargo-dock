# Environment card

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视仓库中的连续控制刚体接触任务：一辆轮式小车需要通过推挤，将一个自由滑动的脆弱箱子从隔墙近侧经窄开口运到远侧的 dock 矩形区域内，并让箱子最终满足“完全进入 dock、朝向与 dock 对齐、速度接近零并持续若干步”的停靠条件。主目标是完成箱子交付并稳定停靠；次目标/约束是避免硬撞击损坏箱子、避免小车或箱子离开仓库地板、避免时间耗尽。不应把“接近箱子”“把箱子推到 dock 附近”“碰到 dock 区域”或“时间耗尽”当作成功。

## 2. 任务类型选择
selected_route_id: manipulation_grasping  
confidence: high  
reason: 核心目标是操控一个可自由移动物体到指定位姿并稳定停靠，虽然执行器不是夹爪而是推挤接触，但仍属于非抓取式 manipulation。硬撞击约束、出界约束、时间限制是约束或附属风险，不是与“把箱子送入 dock”等权且冲突的多目标核心，因此不选 multi_objective_task。也不应选 navigation_goal_reaching，因为小车自身到达不是主目标，箱子到 dock 才是。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 所有条目裁剪到 [-2.0, 2.0]
- 说明：obs[0]/obs[1] 为归一化位置，仓库半宽约为 5.0，半高约为 4.0；obs[6]/obs[7]/obs[8]/obs[9] 的原始长度/速度量纲除以 3.0；obs[12]/obs[13] 为 crate 到 dock 中心的带符号偏移，分别按仓库半宽/半高归一化。

- obs[0]: cart_x，小车 x 位置 / 仓库半宽，0 为中心线，+1 为远墙方向；reward_usable: true
- obs[1]: cart_y，小车 y 位置 / 仓库半高，0 为中心线；reward_usable: true
- obs[2]: cart_cos_heading，小车朝向余弦；reward_usable: true
- obs[3]: cart_sin_heading，小车朝向正弦；reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身朝向速度 / 3.0；reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度 / 8.0；reward_usable: true
- obs[6]: crate_rel_x_body，crate 相对小车在车体坐标系下的 x 分量 / 3.0；reward_usable: true
- obs[7]: crate_rel_y_body，crate 相对小车在车体坐标系下的 y 分量 / 3.0；reward_usable: true
- obs[8]: crate_vx，crate 世界坐标系 x 线速度 / 3.0；reward_usable: true
- obs[9]: crate_vy，crate 世界坐标系 y 线速度 / 3.0；reward_usable: true
- obs[10]: crate_cos_heading，crate 朝向余弦；reward_usable: true
- obs[11]: crate_sin_heading，crate 朝向正弦；reward_usable: true
- obs[12]: crate_to_dock_x，crate 中心到 dock 中心的带符号 x 偏移 / 仓库半宽；reward_usable: true
- obs[13]: crate_to_dock_y，crate 中心到 dock 中心的带符号 y 偏移 / 仓库半高；reward_usable: true
- obs[14]: cart_crate_contact，小车与 crate 当前是否接触，1.0 为接触，0.0 为不接触；reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍 proximity，0 为范围内无碰撞风险，1 为接触；reward_usable: true，但只建议条件使用
- obs[16]: sensor_left，小车左侧最近静态障碍 proximity，0 为范围内无碰撞风险，1 为接触；reward_usable: true，但只建议条件使用
- obs[17]: sensor_right，小车右侧最近静态障碍 proximity，0 为范围内无碰撞风险，1 为接触；reward_usable: true，但只建议条件使用
- obs[18]: time_fraction，已消耗时间比例 [0,1]；reward_usable: true，但初期不建议作为主职责

## 4. 动作空间 action_space
- type: Box
- shape: [2]
- continuous: true
- bounds: [-1.0, 1.0] per channel
- action[0]: drive，沿小车自身朝向的纵向力命令，+1 前进，-1 后退。
- action[1]: steer，转向力矩命令，+1 左转/逆时针，-1 右转/顺时针。
- 重要限制：小车本身没有刹车，也没有夹爪；crate 只能通过接触被推动，小车停止推挤后 crate 仍会因已有速度滑行，仅靠地面阻尼减速。

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination:
  - docked_success：crate 完全在 dock 内，crate 朝向误差小于 30 度，crate 速度小于 0.05 m/s，并连续保持 10 个环境步。
  - 该终止没有显式 success flag 可读；只能通过 obs 间接推断，标注为 derived_possible。
- failure-like termination:
  - crate_out_of_bounds：crate 中心离开仓库地面矩形。
  - cart_out_of_bounds：小车中心离开仓库地面矩形。
  - crate_damaged：crate 受到 3 次或以上硬撞击；硬撞击定义为小车-crate 接触峰值法向冲量超过脆弱性阈值。
  - 这些失败没有显式 failure flag 可读；crate/cart 出界可从位置信号间接推断，硬撞击只能从接触和相对速度粗略代理，标注为 derived_possible。
- ambiguous termination:
  - time_limit：达到固定步数预算，报告为 truncation，不是任务成功。
  - 该截断在 compute_reward 签名中不可直接读取。
- truncation:
  - time_limit 是唯一明确 truncation 模式。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []
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
- 额外不可直接读取项：
  - compute_reward 签名不包含 done/terminated/truncated。
  - dock 的精确矩形尺寸、dock 朝向、隔墙开口几何、硬撞击冲量阈值均未在 obs 中显式给出。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段：当前为无，allowed_info_fields: []
- training_progress：仅当 prompt 明确允许时使用；本任务未明确允许，因此默认不用。

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片
- 被 mask 或 forbidden 的 info 字段，包括但不限于 is_success、cargo_goal_distance、cargo_angle_error、cargo_speed、robot_cargo_distance、contact_impulse、hard_collision_count、stagnation_steps、action_energy、component_returns、official_reward_terms、termination_reason、cargo_inside_dock、stable_steps。
- 不能依赖 done/terminated/truncated 做终端 bonus/penalty，因为该函数签名没有这些参数，info 中也禁止读取终止原因。

## 7. 可用于奖励函数的信号
- position:
  - cart_x: obs[0]
  - cart_y: obs[1]
  - crate 世界坐标：可由 obs[0], obs[1], obs[2], obs[3], obs[6], obs[7] 派生。先用车体朝向旋转 (obs[6]*3.0, obs[7]*3.0)，再加上小车位置 (obs[0]*5.0, obs[1]*4.0)。
  - crate_to_dock offset: obs[12], obs[13]，可直接用于构造 crate 到 dock 的距离或偏移量。
  - 出界判断 derived_possible：cart 出界可通过 |obs[0]| > 1 或 |obs[1]| > 1 推断；crate 出界可通过派生 crate 世界坐标是否超出约 [-5,5] × [-4,4] 推断。
- velocity:
  - cart_forward_speed: obs[4]
  - cart_yaw_rate: obs[5]
  - crate_vx: obs[8]
  - crate_vy: obs[9]
  - crate 线速度大小：sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2) derived_possible
  - 小车世界速度：可由 obs[2], obs[3], obs[4] 派生
  - cart-crate 相对速度：可由小车速度与 crate 速度派生，用于接触柔顺性代理 derived_possible
- orientation:
  - cart_cos_heading: obs[2]
  - cart_sin_heading: obs[3]
  - crate_cos_heading: obs[10]
  - crate_sin_heading: obs[11]
  - crate 朝向角：atan2(obs[11], obs[10]) derived_possible
  - crate 相对 dock 的朝向误差：只有在能假设 dock 朝向与仓库坐标轴对齐时才可间接推断；否则不可用，derived_possible / uncertain。
- contact:
  - cart_crate_contact: obs[14]
  - sensor_front: obs[15]
  - sensor_left: obs[16]
  - sensor_right: obs[17]
  - 硬撞击代理 derived_possible：obs[14] 为 1 时，用小车与 crate 的相对速度大小作为硬撞击风险代理；真实峰值法向冲量不可读。
- action/engine:
  - action[0] drive
  - action[1] steer
  - 动作幅度、动作变化率、动作平滑性可由 action 与上一步 action 派生；但 info["action_energy"] 被禁止。
- other:
  - time_fraction: obs[18]
  - 成功停靠代理 derived_possible：|obs[12]| 与 |obs[13]| 接近 0、crate 速度接近 0、crate 朝向与假设的 dock 朝向对齐、接触可能为 0。无法读取 stable_steps 或 success flag。
  - 时间耗尽代理 derived_possible：obs[18] 接近 1，但 compute_reward 无法知道 episode 是否真的 truncation。

## 8. 不确定或不可用的信号
- info 中所有内部诊断字段均不可用于生成奖励：contact_impulse、hard_collision_count、stagnation_steps、action_energy、cargo_inside_dock、stable_steps、termination_reason、is_success 等。
- 精确硬撞击冲量、脆弱性阈值、硬撞击计数不可直接读取。
- 精确 dock 矩形尺寸、dock 朝向、crate 是否“完全进入”dock 不可直接验证。
- 隔墙开口的精确位置和尺寸不可直接读取，只能通过 proximity 传感器间接感知。
- stable_steps 和成功保持时长不可直接读取。
- 终止类型 done/terminated/truncated 未传入 compute_reward。
- original_reward 和官方 reward 项不可用。
- training_progress 未明确允许，默认不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_nonprehensile_push_transport_fragile
control_type: continuous
morphology:
  body_type: planar wheeled cart
  actuator_type: longitudinal force plus steering torque
  contact_structure: nonprehensile cart-crate contact, freely sliding crate, static partition wall with narrow opening, floor damping, no gripper, no brake
primary_objectives:
  - move fragile crate into dock rectangle
  - align crate heading with dock
  - settle crate to near-zero speed inside dock for a short continuous period
secondary_objectives:
  - avoid damaging crate by repeated hard impacts
  - keep cart and crate inside warehouse floor
  - navigate cart and crate through narrow partition opening
main_failure_risks:
  - three or more hard cart-crate impacts
  - cart leaves warehouse floor
  - crate leaves warehouse floor
  - crate overshoots dock because cart cannot brake it
  - crate reaches dock proximity but never satisfies settling conditions
  - time budget exhausted as truncation
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: crate_transport_progress
  purpose: 奖励把 crate 向 dock 推进的净进展。
  why_required: 主任务是交付 crate；没有这个职责，智能体缺少主要学习方向。
  usable_signals: obs[12], obs[13] 可构造 crate_to_dock 距离；next_obs 同字段可构造 delta。
  risks: 若只用 proximity 状态值，智能体可能把 crate 推到 dock 附近后停住；若只用 delta，需配合 settle 与 gentle 约束，避免高速冲撞和过冲震荡。

- role_id: gentle_contact_guard
  purpose: 抑制 cart-crate 高速接触，降低硬撞击和损坏风险。
  why_required: crate 是 fragile，硬撞击累计 3 次即失败。
  usable_signals: obs[14] contact, obs[4] cart_forward_speed, obs[8]/obs[9] crate velocity, obs[2]/obs[3] cart heading, obs[6]/obs[7] crate relative position；可派生相对速度或接近速度。derived_possible。
  risks: 没有真实 impulse；用相对速度代理可能过严，导致智能体不敢接触或推动 crate。

- role_id: bounds_safety
  purpose: 阻止 cart 或 crate 离开仓库地板。
  why_required: cart 或 crate 出界都会失败终止。
  usable_signals: obs[0], obs[1] 可直接判断 cart 是否接近/超过归一化边界；crate 世界坐标可由 obs[0], obs[1], obs[2], obs[3], obs[6], obs[7] 派生。derived_possible。
  risks: 惩罚范围过大会干扰正常靠墙、通过开口或推箱子；应只在接近边界或超出边界时使用。

### 10.2 条件职责 conditional_roles
- role_id: docking_settling
  condition_to_use: 当 crate 已接近 dock，即 |obs[12]| 与 |obs[13]| 较小时，或训练中观察到 crate 在 dock 附近高速滑行/震荡时。
  usable_signals: obs[8], obs[9] 可算 crate 速度；obs[12], obs[13] 判断是否在 dock 附近；obs[14] 判断是否仍在被推。
  risks: 过早惩罚速度会阻碍把 crate 运到 dock；过晚则无法阻止 crate 因无刹车而滑过 dock。

- role_id: heading_alignment
  condition_to_use: crate 接近 dock 且可合理假设 dock 朝向与仓库坐标轴对齐时。
  usable_signals: obs[10], obs[11] 可算 crate 朝向；若 dock 轴对齐，则可用 atan2(obs[11], obs[10]) 近似朝向误差。derived_possible / uncertain。
  risks: dock 朝向没有显式给出；错误假设会引导 crate 转向错误方向。

- role_id: cart_to_crate_approach
  condition_to_use: 早期阶段，cart 与 crate 距离较大且 crate 尚未向 dock 移动时。
  usable_signals: obs[6], obs[7] 为 crate 在车体坐标下的相对位置，可派生 cart-crate 距离。
  risks: 可能鼓励高速撞击 crate；需与 gentle_contact_guard 联合，或对接近速度做门控。

- role_id: obstacle_avoidance_narrow_passage
  condition_to_use: 当训练显示 cart 反复卡在隔墙、开口或传感器 proximity 持续偏高时。
  usable_signals: obs[15], obs[16], obs[17] proximity 传感器；obs[0], obs[1] cart 位置；obs[12], obs[13] 判断是否已经接近 dock 方向。
  risks: 可能阻止必要的 cart-crate 接触，或阻止智能体进入窄开口；必须条件化、门控化，不宜作为主职责。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_success_bonus
  reason: compute_reward 不接收 done/terminated/truncated，且 info 中 is_success、termination_reason、stable_steps、cargo_inside_dock 被禁止；无法可靠知道成功终止。
  forbidden_or_missing_signals: done, terminated, truncated, info["is_success"], info["termination_reason"], info["stable_steps"], info["cargo_inside_dock"]

- role_id: hard_collision_count_penalty
  reason: info["hard_collision_count"] 与 contact_impulse 被禁止；无法直接计数硬撞击，只能用接触和相对速度做代理。
  forbidden_or_missing_signals: contact_impulse, hard_collision_count

- role_id: action_energy_penalty
  reason: 任务描述未要求节能，且 info["action_energy"] 被禁止；从 action 虽可计算能量，但可能抑制必要的推挤力。
  forbidden_or_missing_signals: action_energy

- role_id: time_penalty
  reason: 任务未要求最快完成；过早加时间惩罚会鼓励冲撞 fragile crate。obs[18] 可用，但应作为后期条件职责，而不是主职责。
  forbidden_or_missing_signals: 无必须缺失信号；但建议初期避免。

- role_id: stagnation_penalty
  reason: info["stagnation_steps"] 被禁止；虽可由 crate_to_dock delta 粗略推断，但与运输进展职责可能重复且易误伤早期探索。
  forbidden_or_missing_signals: stagnation_steps

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_transport_progress | obs[12], obs[13]；next_obs 同字段；可派生 crate_to_dock_distance | 无 dock 尺寸也可算中心距离；完全 inside 条件不可直接验证 | delta_distance, improvement, potential_based_shaping | 主信号应关注 crate 到 dock 中心的净减少；不能只奖励 proximity，否则易悬停。 |
| gentle_contact_guard | obs[14], obs[4], obs[8], obs[9], obs[2], obs[3], obs[6], obs[7]；可派生相对速度 | contact_impulse, hard_collision_count | hinge_penalty, bounded_signal, gating | 真实硬撞击不可读，只能用接触时相对速度代理，标注 derived_possible。 |
| bounds_safety | obs[0], obs[1]；派生 crate 世界坐标 | 精确仓库边界虽未显式给出，但可由归一化约定近似 | hinge_penalty, bounded_signal | cart 出界可由 abs(obs[0])>1 或 abs(obs[1])>1 推断；crate 出界由派生位置推断。 |
| docking_settling | obs[8], obs[9], obs[12], obs[13], obs[14] | stable_steps, cargo_inside_dock | speed_penalty, bounded_signal, gating, exponential_decay | 条件职责：只在 crate 接近 dock 后启用；用于对抗无刹车滑行和过冲。 |
| heading_alignment | obs[10], obs[11]；若 dock 轴对齐则派生 heading error | dock 朝向未显式给出 | angle_error_penalty, cos_similarity, bounded_signal | 条件职责，且依赖 dock 朝向假设；不确定时不应强制使用。 |
| cart_to_crate_approach | obs[6], obs[7]；派生 cart-crate 距离 | 无 | delta_distance, improvement | 早期引导 cart 接近 crate；需用 gentle_contact_guard 或低速门控防止冲撞。 |
| obstacle_avoidance_narrow_passage | obs[15], obs[16], obs[17], obs[0], obs[1] | 隔墙精确几何、开口位置 | proximity_penalty, bounded_signal, gating | 仅在卡墙/撞墙成为训练问题时启用；不宜默认惩罚所有墙面接近，否则可能阻止进入开口。 |
| terminal_success_bonus | 无可靠终端信号 | done, terminated, truncated, is_success, termination_reason, stable_steps | 无 | 禁用；compute_reward 无法可靠读取成功终止。 |
| hard_collision_count_penalty | 仅可代理：obs[14], 相对速度 | contact_impulse, hard_collision_count | 无直接可用算子 | 禁用直接计数惩罚；改用 gentle_contact_guard。 |
| action_energy_penalty | action[0], action[1] | action_energy（info 禁止） | quadratic_penalty, absolute_penalty | 非任务要求，初期禁用或极弱条件使用。 |
| time_penalty | obs[18] | 无 | linear_penalty, bounded_signal | 初期禁用；后期若严重超时才弱条件加入，避免鼓励冲撞。 |
| stagnation_penalty | 可由 obs[12], obs[13] 的 delta 粗略推断 | stagnation_steps | hinge_penalty, bounded_signal | 初期禁用；与 crate_transport_progress 可能重复。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 小车高速撞击 crate，导致硬撞击失败 | 训练日志中 hard_collision_count 增长；obs[14]=1 时派生相对速度偏高；episode 提前失败 | 强化 gentle_contact_guard；当接近 crate 时对高速接触做 hinge 惩罚或门控；降低接近阶段速度奖励。 |
| crate 被推过 dock 或因无刹车滑出停靠区 | obs[12]/obs[13] 在 0 附近反复穿越；crate 速度在 dock 附近仍高；稳定停靠失败 | 加入 docking_settling，在 crate 接近 dock 后惩罚速度、奖励低速保持；减少 dock 附近的推进奖励。 |
| crate 到达 dock 附近但不停稳 | crate 速度不趋近 0；obs[12]/obs[13] 长期接近 0 但 episode 未成功 | 在接近 dock 后条件化使用 crate 速度惩罚；用 bounded_signal 奖励低速；避免只奖励 proximity。 |
| 小车或 crate 出界 | cart 的 abs(obs[0]) 或 abs(obs[1]) 接近/超过 1；派生 crate 世界坐标接近/超过仓库边界 | 加入 bounds_safety；在接近边界时快速增强惩罚；门控运输奖励，避免把 crate 推向边界。 |
| 小车卡在隔墙或无法通过窄开口 | obs[15]/obs[16]/obs[17] 持续高；cart 位置停滞；crate 未向 dock 移动 | 条件加入 obstacle_avoidance_narrow_passage；可能需要靠近开口方向 shaping；检查是否惩罚所有墙面接触过度。 |
| 小车只接近 crate 但不推动 | obs[6]/obs[7] 距离小但 obs[12]/obs[13] 几乎不变；obs[14] 频繁为 1 但 crate 速度低 | 检查 cart_to_crate_approach 是否过强导致悬停；主职责应仍以 crate 到 dock 的 delta 为主。 |
| 主奖励被 dock 附近悬停收割 | crate 长期在 dock 附近但速度不为 0，episode 以 truncation 结束 | 避免纯 proximity 主信号；用 delta 作为运输进展；接近 dock 后转为 settling 职责。 |
| crate 朝向不符合 dock 要求 | 接近 dock 时 obs[10]/obs[11] 对应的朝向与 dock 轴明显偏离 | 若可推断 dock 朝向，条件加入 heading_alignment；否则先检查 dock 朝向是否可从成功终止数据中间接估计。 |
| 智能体不敢接触 crate | crate 几乎不动；cart 在 crate 附近徘徊；接触时间极短 | 检查 gentle_contact_guard 是否过严；可对低速接触放宽惩罚，只惩罚高速相对接近。 |
| 时间预算耗尽但未完成 | obs[18] 接近 1；crate 未满足稳定停靠代理条件 | 后期可弱加 time_penalty 或效率 shaping；但不能早于 gentle_contact_guard 和 transport_progress 稳定。 |



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
# ============================================================================
# 自检记录（单步奖励估算，单位：奖励/步）
#   几何：仓库半宽 5.0 m，半高 4.0 m；obs[6]/obs[7]/obs[8]/obs[9] 除以 3.0；
#         obs[12]/obs[13] 分别按半宽/半高归一化。
#
#   R_idle    ≈ -0.002       （无接触、无位移：只有 time_cost）
#   R_push    ≈ +0.09        （正常推行：progress≈+0.05, approach_cargo≈+0.05,
#                              roughness≈-0.01, time+action≈-0.003）
#   R_settled ≈ +1.997       （泊位内 + 对齐 + 慢：settle 每步 +2.0 再扣成本）
#
#   自检①  idle(-0.002) < push(+0.09)                → 严格成立 ✓
#   自检②  泊位外 0.3m 悬停 400 步 ≈ -0.8
#          真正入泊并停稳 ≈ +18(停稳 9 步) + 20(第 10 步被裁剪的完成事件) ≈ +38
#          +38 > -0.8                                 → 严格成立 ✓
#   自检③  closing=1.00 → roughness = -0.30, hard_hit = 0（>1.0 才触发）
#          closing=0.05 → roughness = -0.015
#          差值 0.285 ≫ 正常推箱单步推进值(≈0.05~0.10)，约 3~6 倍       ✓
#          且该惩罚在 closing≈0 的"匀速正常推箱"时为 0，不压制推进。
#   自检④  同一"停稳"状态连续调用 12 次，每步差值恒为 +2.0（完成事件只在
#          第 10 步叠加一次），停稳收益不会被任何开关关掉。            ✓
#   自检⑤  R_push - R_idle ≈ 0.09，明显大于正常单步罚项量级
#          (roughness≈0.01, time_cost 0.002, action_cost 0.0005)       ✓
#          R_settled > R_push                                          ✓
#          量级：R_push-R_idle 与推进项同量级（都是 1e-1），非千分之一。 ✓
#   越界自检：|obs[0]|=1.05 → bounds_safety = -2.55，远低于场地中央的 0。  ✓
#
#   完成谓词（显式从 obs 推断，容差取自环境事实）：
#     |crate→dock| < 0.25 m 且 |crate 朝向 cos| >= 0.866 (30°) 且
#     crate 速度 < 0.05 m/s，连续 10 步成立 → 一次性 +300。
#   停稳每步收益与完成谓词使用同一组硬门控，故不可能"半对齐赖着收分"。
# ============================================================================

_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_PAID_FAIL = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------------- 0. 回合边界检测（obs[18] 单调递增，重置时回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _PAID_FAIL[0] = False
    _PREV_T[0] = t

    # ---------------- 1. 由 obs 还原几何量（单位：米） ----------------
    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    c = obs[2]
    s = obs[3]

    # 货箱在小车车体坐标系下的相对位置
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    d_cart_crate = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = next_obs[6] * 3.0
    nrel_y = next_obs[7] * 3.0
    nd_cart_crate = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    # 货箱中心 -> 泊位中心距离
    ddx = obs[12] * 5.0
    ddy = obs[13] * 4.0
    d_crate_dock = (ddx * ddx + ddy * ddy) ** 0.5

    nddx = next_obs[12] * 5.0
    nddy = next_obs[13] * 4.0
    nd_crate_dock = (nddx * nddx + nddy * nddy) ** 0.5

    # 货箱世界坐标（用于越界守卫）
    crate_x = cart_x + c * rel_x - s * rel_y
    crate_y = cart_y + s * rel_x + c * rel_y

    # ---------------- 2. 分项实现 ----------------

    # 2.1 approach_cargo：+1.0 / 米，本帧小车->货箱距离缩短量（有符号、对称）
    approach_cargo = 1.0 * (d_cart_crate - nd_cart_crate)

    # 2.2 progress：+1.0 / 米，本帧货箱->泊位距离缩短量（有符号、对称）
    progress = 1.0 * (d_crate_dock - nd_crate_dock)

    # 2.3 dock_enter：+5.0，货箱首次完全进入泊位容差，整局只发一次
    in_dock = 1.0 if nd_crate_dock < 0.25 else 0.0
    dock_enter = 0.0
    if in_dock > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # 2.4 roughness：接触冲量比例的可观测代理（接近速度 x 接触）
    #     只在"接触且正在接近"时为负；匀速推箱（closing≈0）不受罚。
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.30 * contact * closing

    # 2.5 action_cost：-0.0005 * 动作平方和
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])

    # 2.6 time_cost：-0.002 / 步
    time_cost = -0.002

    # 2.7 hard_hit：单步硬冲击（接触 + 高速接近）固定惩罚 + 计数
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] += 1

    # 2.8 完成侧状态量（全部来自 next_obs）
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align_cos = next_obs[10]
    if align_cos < 0.0:
        align_cos = -align_cos

    align_ok = 1.0 if align_cos >= 0.866 else 0.0     # 朝向误差 < 30 度
    slow_ok = 1.0 if crate_speed < 0.05 else 0.0      # 速度 < 0.05 m/s
    # 停稳期每步收益：硬门控（不满足完成谓词一分不给，不存在"半对齐刷分"），
    # 且不因任何"已发放"开关而停止。
    settle_bonus = 2.0 * in_dock * align_ok * slow_ok

    # 2.9 terminal_success：完成谓词连续 10 步成立，整局只发一次
    if in_dock > 0.5 and align_ok > 0.5 and slow_ok > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 300.0

    # 2.10 越界度量（小车按 obs 归一化，货箱按世界坐标 / 半宽、半高）
    cart_margin = abs(obs[0])
    if abs(obs[1]) > cart_margin:
        cart_margin = abs(obs[1])
    crate_margin = abs(crate_x) / 5.0
    if abs(crate_y) / 4.0 > crate_margin:
        crate_margin = abs(crate_y) / 4.0

    # 2.11 terminal_failure：越界 或 累计硬冲击 >= 3，整局只发一次
    terminal_failure = 0.0
    if not _PAID_FAIL[0]:
        if cart_margin > 1.05 or crate_margin > 1.0 or _HARD_HITS[0] >= 3:
            _PAID_FAIL[0] = True
            terminal_failure = -100.0

    # 2.12 bounds_safety：随越界接近单调下降的守卫惩罚
    m = cart_margin
    if crate_margin > m:
        m = crate_margin
    bounds_safety = 0.0
    if m > 0.90:
        over = m - 0.90
        bounds_safety = -(2.0 * over + 100.0 * over * over)

    # ---------------- 3. 汇总 ----------------
    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settle_bonus": settle_bonus,
        "bounds_safety": bounds_safety,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }

    total = (approach_cargo + progress + dock_enter + roughness + action_cost
             + time_cost + hard_hit + settle_bonus + bounds_safety
             + terminal_success + terminal_failure)

    return (float(total), components)
```

# Reward reflection of the current reward (native task score = 3.9019)
### Task score

- mean_eval_reward: 3.901935094269054
- mean_episode_length: 400
- eval episodes: 20
- termination breakdown: {'terminated': 0, 'truncated': 20}

### Episode return during training

| training progress | mean episode return | mean episode length |
|---|---:|---:|
| 17% | 219.88 | 393.5 |
| 33% | 218.14 | 392.8 |
| 50% | 217.64 | 394.4 |
| 67% | 221.58 | 393.0 |
| 83% | 232.58 | 393.8 |
| 100% | 220.83 | 394.3 |

### Reward component values during training (mean reward per episode)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | -0.455 | 0.310 | 7.813 | 125.870 | 283.350 | 336.699 | 376.051 | 355.327 | 353.906 | 376.312 | 410.656 |
| settle_bonus | 0.000 | 0.189 | 3.596 | 107.410 | 263.602 | 315.663 | 353.748 | 333.856 | 332.876 | 354.136 | 386.667 |
| terminal_success | 0.000 | 1.200 | 19.947 | 191.474 | 209.302 | 227.059 | 245.020 | 233.378 | 232.889 | 244.000 | 266.667 |
| dock_enter | 0.287 | 0.473 | 1.469 | 4.026 | 4.063 | 4.235 | 4.482 | 4.428 | 4.234 | 4.407 | 4.630 |
| progress | 0.660 | 1.986 | 3.005 | 3.487 | 3.711 | 3.736 | 3.742 | 3.763 | 3.631 | 3.712 | 3.782 |
| approach_cargo | 0.893 | 1.214 | 1.148 | 1.125 | 1.034 | 0.966 | 0.887 | 0.885 | 0.846 | 0.863 | 0.991 |
| time_cost | -0.800 | -0.799 | -0.796 | -0.731 | -0.773 | -0.786 | -0.795 | -0.798 | -0.799 | -0.800 | -0.800 |
| roughness | -0.405 | -1.245 | -1.158 | -0.729 | -0.544 | -0.486 | -0.432 | -0.450 | -0.430 | -0.389 | -0.398 |
| action_cost | -0.201 | -0.217 | -0.220 | -0.208 | -0.218 | -0.227 | -0.226 | -0.220 | -0.220 | -0.222 | -0.218 |
| terminal_failure | -0.933 | -2.000 | -0.931 | 0.000 | -0.258 | -0.131 | -0.266 | -0.532 | -0.399 | -0.133 | 0.000 |
| bounds_safety | -0.713 | -0.978 | -0.236 | -0.001 | -0.033 | 0.000 | 0.000 | -0.033 | -0.129 | -0.005 | 0.000 |
| hard_hit | -0.001 | -0.009 | -0.015 | -0.001 | -0.006 | -0.005 | -0.007 | -0.008 | -0.004 | -0.007 | 0.000 |

### Reward component activation rate during training (fraction of steps where the component is non-zero)

| component | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% | 100% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| total_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settle_bonus | 0.0% | 0.0% | 0.5% | 14.7% | 34.1% | 40.1% | 44.5% | 41.9% | 41.7% | 44.3% | 45.5% |
| terminal_success | 0.0% | 0.0% | 0.0% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| dock_enter | 0.0% | 0.0% | 0.1% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% |
| progress | 34.7% | 77.7% | 82.4% | 68.7% | 46.9% | 45.1% | 43.4% | 43.3% | 41.4% | 41.1% | 40.5% |
| approach_cargo | 100.0% | 99.9% | 99.9% | 99.8% | 99.8% | 99.8% | 99.7% | 99.7% | 99.8% | 99.6% | 99.6% |
| time_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| roughness | 4.1% | 15.9% | 15.5% | 15.1% | 13.1% | 12.6% | 12.6% | 11.9% | 11.5% | 12.4% | 13.4% |
| action_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| terminal_failure | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| bounds_safety | 0.3% | 0.3% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% |
| hard_hit | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

### Reward component values (episode sums over all training episodes)

| component | mean | abs mean | min | max |
|---|---:|---:|---:|---:|
| action_cost | -0.2179 | 0.2179 | -0.3282 | -0.0969 |
| approach_cargo | 0.9874 | 1.0095 | -2.7705 | 2.4143 |
| bounds_safety | -0.2089 | 0.2089 | -108.7633 | 0.0000 |
| dock_enter | 3.2283 | 3.2283 | 0.0000 | 5.0000 |
| hard_hit | -0.0061 | 0.0061 | -3.5000 | 0.0000 |
| progress | 3.1521 | 3.1624 | -3.6684 | 4.4990 |
| roughness | -0.6264 | 0.6264 | -6.0840 | 0.0000 |
| settle_bonus | 206.6716 | 206.6716 | 0.0000 | 504.0000 |
| terminal_failure | -0.5494 | 0.5494 | -100.0000 | 0.0000 |
| terminal_success | 161.3996 | 161.3996 | 0.0000 | 300.0000 |
| time_cost | -0.7873 | 0.7873 | -0.8000 | -0.3400 |
| total_reward | 221.7752 | 222.4676 | -122.5756 | 530.2405 |