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