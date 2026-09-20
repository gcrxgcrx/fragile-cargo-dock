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