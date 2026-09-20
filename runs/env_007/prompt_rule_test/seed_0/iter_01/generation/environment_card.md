# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只**易碎**的方形货箱从仓库近侧推到隔墙另一侧的**交付泊位**内，并让货箱在泊位中**完全进入、朝向对齐、几乎静止**并持续一小段时间。主目标是“把货箱安全送达并稳定停靠”；次目标是“避免货箱受到硬冲击损坏”“避免小车或货箱离开仓库地面”“在时间预算内完成”。**不该混淆的目标**：单纯“靠近泊位”“触碰泊位区域”“把货箱推得越快越好”都不是成功；由于小车无法从后方减速货箱，把“持续推到底”当作目标会与低速停靠条件冲突。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: medium
reason: 核心是“把物体（货箱）移动到指定位姿（泊位内、朝向对齐、静止）”，属于对物体的操控与位姿达成，而非单纯导航或持续前进。虽然载体是轮式小车、动作是驾驶式连续控制，但成功判据完全落在**货箱的位姿与速度**上，且存在“接近—推入—稳定停靠”的阶段结构，因此归为 manipulation_grasping 最贴切。次目标（不损坏、不出界、限时）是约束而非并列主目标，故不选 multi_objective_task。

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
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=远，1=接触），reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例 [0,1]，reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [2]
- bounds: 每通道 [-1.0, 1.0]
- action[0]: drive，沿小车朝向的纵向力指令；+1 前进，-1 倒车
- action[1]: steer，转向力矩指令；+1 左转（逆时针），-1 右转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `docked_success` —— 货箱完全在泊位内、朝向误差 < 30°、速度 < 0.05 m/s，且连续保持 10 个环境步。
- failure-like termination: `crate_out_of_bounds`（货箱中心离开仓库地面）、`cart_out_of_bounds`（小车中心离开仓库地面）、`crate_damaged`（硬冲击计数 ≥ 3）。
- ambiguous termination: 无显式歧义终止；但“货箱进入泊位但未满足朝向/速度/持续条件”不会终止，属于未完成状态。
- truncation: `time_limit` —— 达到固定步数预算，报告为截断，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被列为 forbidden）
- explicit_failure_flag_available: false（`termination_reason`、`hard_collision_count` 等被 forbidden）
- allowed_info_fields: []（无任何允许读取的 info 字段）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

> 说明：成功/失败只能通过观测信号**间接推断**（derived_possible），不能直接读取 info。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，19 维）
- action（当前动作，2 维）
- next_obs（下一步观测，19 维）
- info 中明确允许的字段：**无**（allowed_info_fields 为空）
- training_progress：仅当 prompt 明确允许时才用；本环境未声明允许，默认不使用

禁止使用：
- original_reward（官方奖励被 mask）
- official_reward / masked_reward
- 任何未声明的 info 字段（尤其 5.2 列出的 forbidden 字段）
- 未声明的 obs 切片（只能使用 0–18 已声明维度）

## 7. 可用于奖励函数的信号
- position:
  - 小车世界位置：由 obs[0]*5.0, obs[1]*4.0 恢复（仓库半宽 5.0，半高 4.0）。
  - 货箱世界位置：将 (obs[6]*3.0, obs[7]*3.0) 按小车朝向旋转后加上小车位置。
  - 货箱到泊位偏移：obs[12]*5.0, obs[13]*4.0（直接可用）。
- velocity:
  - 小车前向速度：obs[4]*3.0。
  - 小车角速度：obs[5]*8.0。
  - 货箱世界速度：obs[8]*3.0, obs[9]*3.0；货箱轴向速率 = sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)。
- orientation:
  - 小车朝向：atan2(obs[3], obs[2])。
  - 货箱朝向：atan2(obs[11], obs[10])；货箱朝向误差可结合泊位朝向推断（derived_possible）。
- contact:
  - 小车-货箱接触：obs[14]（1/0）。
  - 静态障碍接近度：obs[15]/obs[16]/obs[17]（隔墙与开口为硬障碍）。
- action/engine:
  - drive=action[0]，steer=action[1]；可用于动作平滑/能耗类信号（若任务要求）。
- other:
  - time_fraction=obs[18]，可用于时间相关调度或截断前行为调整。
  - derived_possible 推断路径：
    - 成功接近：货箱到泊位偏移 (obs[12], obs[13]) 持续减小且货箱速率趋近 0。
    - 货箱损坏风险：obs[14]=1 且小车前向速度 obs[4] 较大（高冲击代理），或货箱速度突变（derived_possible）。
    - 出界：小车/货箱世界坐标超出合理范围（derived_possible）。
    - 卡住/停滞：货箱到泊位偏移长时间不减小（derived_possible）。

## 8. 不确定或不可用的信号
- 官方奖励 masked_reward：不可用。
- info 全部字段：不可用（含 is_success、termination_reason、hard_collision_count、contact_impulse、cargo_inside_dock、stable_steps 等）。
- 精确的“硬冲击计数”“峰值法向冲量”：不可直接读取，只能用 obs[14] 与速度量做代理（derived_possible，噪声大）。
- 泊位矩形的精确边界与朝向：未在 obs 中显式给出，只能通过 obs[12]/obs[13] 的偏移与货箱朝向间接推断（derived_possible）。
- 隔墙开口的精确几何：未显式给出，只能通过 obs[15]/obs[16]/obs[17] 接近度间接感知。
- 货箱“完全进入泊位”的布尔量：不可直接读取，需由偏移与货箱尺寸推断（derived_possible）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation
control_type: continuous
morphology:
  body_type: wheeled_cart_top_down
  actuator_type: longitudinal_force_plus_steering_torque
  contact_structure: rigid_body_contact_no_gripper_push_only
primary_objectives:
  - 将易碎货箱推入泊位并使其完全进入、朝向对齐、几乎静止并持续保持
secondary_objectives:
  - 避免货箱受到硬冲击（易碎约束）
  - 避免小车或货箱离开仓库地面
  - 在时间预算内完成
main_failure_risks:
  - 硬冲击累计达 3 次导致货箱损坏
  - 小车或货箱出界
  - 货箱进入泊位但速度/朝向不满足，无法稳定停靠
  - 小车无刹车，货箱滑行过头冲出泊位
  - 隔墙开口狭窄，推箱路径被卡住或撞墙
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向泊位中心靠近（接近阶段主信号）。
  why_required: 任务核心是把货箱移动到泊位，必须有指向泊位的进度信号。
  usable_signals: [obs[12], obs[13], obs[6], obs[7], obs[0], obs[1], obs[2], obs[3]]
  risks: 若用“接近度”而非“增量”，agent 可能停在泊位附近不完成；需配合停靠条件。
- role_id: crate_settle_and_align
  purpose: 在货箱接近泊位后，鼓励其低速、朝向对齐并稳定停靠。
  why_required: 成功判据要求货箱完全进入、朝向对齐、速度 < 0.05 m/s 并持续 10 步。
  usable_signals: [obs[8], obs[9], obs[10], obs[11], obs[12], obs[13]]
  risks: 过早强调低速会抑制接近进度；需按阶段或按接近程度条件化。

### 10.2 条件职责 conditional_roles
- role_id: fragile_impact_avoidance
  condition_to_use: 当 obs[14]=1（接触）且小车前向速度 obs[4] 较大时，抑制高冲击推撞。
  usable_signals: [obs[14], obs[4], obs[8], obs[9]]
  risks: 无精确冲量信号，只能用速度代理，可能误罚正常推箱。
- role_id: boundary_avoidance
  condition_to_use: 当小车或货箱接近仓库边界（由世界坐标推断）时加入。
  usable_signals: [obs[0], obs[1], obs[6], obs[7], obs[2], obs[3]]
  risks: 边界几何未显式给出，需谨慎设定阈值。
- role_id: obstacle_avoidance
  condition_to_use: 当 obs[15]/obs[16]/obs[17] 接近 1（接近隔墙/障碍）时加入。
  usable_signals: [obs[15], obs[16], obs[17]]
  risks: 可能抑制必要的穿墙开口操作，需与开口通过行为协调。
- role_id: action_smoothness_or_energy
  condition_to_use: 仅当任务明确要求高效/平滑时加入；本任务描述未强调，默认不加。
  usable_signals: [action[0], action[1]]
  risks: 属附属优化，可能干扰主任务。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用。
  forbidden_or_missing_signals: [masked_reward, original_reward, official_reward_terms]
- role_id: info_based_success_bonus
  reason: info 全部字段 forbidden，无法读取 is_success/termination_reason/cargo_inside_dock/stable_steps。
  forbidden_or_missing_signals: [is_success, termination_reason, cargo_inside_dock, stable_steps]
- role_id: hard_collision_penalty_from_info
  reason: hard_collision_count/contact_impulse 被 forbidden，只能用速度代理。
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse]
- role_id: push_until_end
  reason: 小车无刹车，货箱靠地面阻尼滑行；持续推到底会破坏低速停靠条件。
  forbidden_or_missing_signals: []
- role_id: pure_proximity_hover
  reason: 仅用接近度会让 agent 停在泊位附近收割分数而不完成停靠。
  forbidden_or_missing_signals: []

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13]（及恢复的世界坐标） | 无 | delta(distance), improvement | 用增量而非纯接近度，避免悬停 |
| crate_settle_and_align | obs[8], obs[9], obs[10], obs[11], obs[12], obs[13] | 泊位精确朝向/边界 | bounded_signal, quadratic_penalty, hinge | 按接近程度条件化，避免抑制接近 |
| fragile_impact_avoidance | obs[14], obs[4], obs[8], obs[9] | contact_impulse, hard_collision_count | hinge, gate | 速度代理，阈值需保守 |
| boundary_avoidance | obs[0], obs[1], obs[6], obs[7], obs[2], obs[3] | 精确边界几何 | hinge, quadratic_penalty | derived_possible，阈值需推断 |
| obstacle_avoidance | obs[15], obs[16], obs[17] | 开口精确几何 | hinge, gate | 避免误罚穿开口行为 |
| action_smoothness_or_energy | action[0], action[1] | 无 | quadratic_penalty | 默认不加，除非明确要求 |
| official_reward_shaping | 无 | masked_reward | — | 禁用 |
| info_based_success_bonus | 无 | is_success 等 | — | 禁用 |
| hard_collision_penalty_from_info | 无 | hard_collision_count | — | 禁用 |
| push_until_end | 无 | 无 | — | 禁用，与停靠条件冲突 |
| pure_proximity_hover | obs[12], obs[13] | 无 | proximity | 慎用，易悬停 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱被推过头冲出泊位 | 货箱到泊位偏移先减小后反向增大；货箱速率在泊位附近仍高 | 在接近泊位时降低推进力度，加入低速/停靠条件化信号 |
| 货箱进入泊位但速度不达标 | 偏移小但货箱速率长期 > 0.05 m/s | 强化低速停靠职责，弱化接近阶段信号 |
| 货箱朝向未对齐 | 货箱朝向误差长期 > 30° | 加入朝向对齐信号，或调整推箱接触点策略 |
| 硬冲击导致货箱损坏 | 接触时小车前向速度大；episode 提前终止 | 加入冲击抑制（hinge/gate），降低推撞速度 |
| 小车或货箱出界 | 世界坐标接近/超出仓库范围 | 加入边界避让信号 |
| 卡在隔墙开口 | 小车/货箱位置长时间不变，接近度传感器高 | 加入障碍避让与开口通过引导 |
| 悬停在泊位附近不完成 | 偏移小但无停靠进展，episode 拖到截断 | 用增量信号替代纯接近度，加入停靠完成条件 |
| 时间耗尽截断 | time_fraction 接近 1 且未停靠 | 调整时间相关调度或提高接近效率 |