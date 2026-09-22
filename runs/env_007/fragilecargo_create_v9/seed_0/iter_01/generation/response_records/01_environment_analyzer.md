# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个俯视视角的仓库非预hensile操作任务：一辆只能施加纵向驱动力和转向力矩、没有刹车、没有夹爪的轮式小车，需要把地面上的一个可自由滑动且易碎的方形货箱，通过推撞方式从隔墙近侧运送到远侧的交付坞。主目标是让货箱完全进入坞内、货箱朝向与坞对齐、货箱接近静止，并持续一小段稳定时间；次目标包括避免货箱受到重复硬冲击、避免小车或货箱离开仓库地面、避免撞墙卡死。不该混淆的目标：小车自身到达坞、货箱仅仅碰到坞区域、货箱高速冲过坞、用持续推挤阻止货箱滑动，这些都不等于任务成功。小车无法从后方给货箱刹车，货箱只能靠地面阻尼减速。

## 2. 任务类型选择
selected_route_id: manipulation_grasping

confidence: high

reason: 核心对象是货箱，成功判据落在货箱位姿、速度、朝向和持续稳定上，而不是小车自身到达某点。虽然这里没有夹爪，但本质是“操控物体到指定位姿”的非预hensile推箱任务，且有阶段目标：接近、推动、穿越开口、对位、稳定停靠。附属的边界约束、易碎约束和时间限制不是与主目标权重相当的核心多目标，因此不选 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 全部裁剪到 [-2.0, 2.0]；部分维度是归一化量，极值可能饱和。
- obs[0]: cart_x，小车 x 位置 / 仓库半宽（5.0），0 为中线，+1 为远墙侧；reward_usable: true
- obs[1]: cart_y，小车 y 位置 / 仓库半高（4.0），0 为中线；reward_usable: true
- obs[2]: cart_cos_heading，小车朝向余弦；reward_usable: true
- obs[3]: cart_sin_heading，小车朝向正弦；reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身朝向速度 / 3.0 m/s；reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度 / 8.0 rad/s；reward_usable: true
- obs[6]: crate_rel_x_body，货箱相对小车在车体坐标系下的 x 分量 / 3.0 m；reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体坐标系下的 y 分量 / 3.0 m；reward_usable: true
- obs[8]: crate_vx，货箱世界系 x 速度 / 3.0 m/s；reward_usable: true
- obs[9]: crate_vy，货箱世界系 y 速度 / 3.0 m/s；reward_usable: true
- obs[10]: crate_cos_heading，货箱朝向余弦；reward_usable: true
- obs[11]: crate_sin_heading，货箱朝向正弦；reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到坞中心的有符号 x 偏移 / 仓库半宽（5.0）；reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到坞中心的有符号 y 偏移 / 仓库半高（4.0）；reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱当前是否接触，1.0 为接触，0.0 为不接触；reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度，0 为范围内清晰，1 为接触；reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，0 为范围内清晰，1 为接触；reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，0 为范围内清晰，1 为接触；reward_usable: true
- obs[18]: time_fraction，已消耗 episode 时间预算比例，范围 [0,1]；reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [2]
- continuous: true
- bounds: [-1.0, 1.0] per channel
- action 0: drive，沿小车朝向的纵向力命令；+1 前进，-1 倒退。
- action 1: steer，转向力矩命令；+1 左转/逆时针，-1 右转。
- 无刹车动作，无夹爪动作。小车无法直接制动货箱，货箱只靠地面阻尼减速。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - docked_success：货箱完全进入坞内，货箱朝向误差小于 30 度，货箱速度小于 0.05 m/s，并且这些条件连续保持 10 个环境步。
  - “完全进入坞内”的精确几何：坞为 0.84 m × 0.84 m，货箱为 0.60 m × 0.60 m，因此要求 |crate_to_dock_x| <= 0.12 m 且 |crate_to_dock_y| <= 0.12 m，即 |obs[12]| <= 0.024 且 |obs[13]| <= 0.030。
  - 成功终止立即结束，之后没有额外时间累积奖励。
- failure-like termination:
  - crate_out_of_bounds：货箱中心离开仓库地面矩形。
  - cart_out_of_bounds：小车中心离开仓库地面矩形。
  - crate_damaged：货箱承受 3 次或更多硬冲击；硬冲击定义为小车-货箱接触的峰值法向冲量超过易碎阈值。
- ambiguous termination:
  - 仅有 episode 结束但无法从 info 读取原因时，需要通过 obs 间接推断，不是显式可读信号。
- truncation:
  - time_limit：达到固定步数预算；报告为截断，不是任务成功。

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

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段：无，当前 allowed_info_fields = []
- training_progress：当前 prompt 未明确允许，禁止使用

禁止使用：
- original_reward
- official_reward
- 任何未声明的 info 字段
- 任何未声明的 obs 切片
- training_progress
- 依赖 info 的终止原因、成功标记、硬冲击计数、冲量、静止步数、坞内标记等内部诊断量

## 7. 可用于奖励函数的信号
- position:
  - obs[0], obs[1]：小车归一化位置，可用于边界约束。
  - obs[12], obs[13]：货箱到坞中心的归一化偏移，可直接推导货箱到坞距离和是否完全进入坞。
  - obs[6], obs[7] + obs[0], obs[1], obs[2], obs[3]：可恢复货箱世界坐标；用于货箱出界推断、相对位置分析。
  - derived_possible：货箱是否完全进入坞内，可由 |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 推断。
  - derived_possible：货箱或小车是否接近/越过仓库边界，可由 obs[0], obs[1] 以及恢复出的货箱世界坐标推断。
- velocity:
  - obs[4]：小车纵向速度，可用于接触冲击代理、推动阶段控制。
  - obs[5]：小车角速度，可用于转向稳定性分析。
  - obs[8], obs[9]：货箱世界系速度，可计算货箱速度大小 sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)。
  - derived_possible：货箱低速稳定条件，可用货箱速度 < 0.05 m/s 推断。
  - derived_possible：小车-货箱相对速度/碰撞烈度代理，可用 obs[14] 接触标记结合 obs[4], obs[8], obs[9] 推断，但无法恢复真实峰值法向冲量。
- orientation:
  - obs[2], obs[3]：小车朝向，用于坐标变换。
  - obs[10], obs[11]：货箱朝向；接口说明中货箱朝向误差代理为 atan2(obs[11], obs[10])。
  - derived_possible：货箱朝向误差是否小于 30 度，可由 atan2(obs[11], obs[10]) 推断。
- contact:
  - obs[14]：小车-货箱是否接触。
  - obs[15], obs[16], obs[17]：静态障碍接近度，可用于隔墙/边界避碰。
  - derived_possible：硬冲击事件代理，可由接触发生时刻的相对速度/货箱速度突变推断，但阈值和计数不可精确读取。
- action/engine:
  - action[0]：纵向力命令，可用于平滑/推力控制，但当前任务没有明确能耗或动作幅度要求。
  - action[1]：转向命令，可用于转向平滑控制。
- other:
  - obs[18]：时间预算消耗比例，可用于时间效率或截断风险分析。
  - derived_possible：成功终止可间接推断为货箱完全进入坞、朝向对齐、速度极低并持续若干步；但持续步数需要内部计数器，不能从单帧 obs 直接读取。

## 8. 不确定或不可用的信号
- info 中所有字段均不可用：is_success、cargo_goal_distance、cargo_angle_error、cargo_speed、robot_cargo_distance、contact_impulse、hard_collision_count、stagnation_steps、action_energy、component_returns、official_reward_terms、termination_reason、cargo_inside_dock、stable_steps。
- original_reward 被 mask，不可用。
- 真实峰值法向冲量不可读；只能构造相对速度/接触代理，不能精确判断“硬冲击”阈值。
- 硬冲击计数不可读；无法直接知道已发生几次 damage。
- 官方易碎冲量阈值不可读。
- 精确静止持续步数不可读；只能通过跨 step 自行维护派生条件。
- 静态障碍传感器的接近度不是精确距离，也不是碰撞冲量。
- 仓库边界外的真实坐标会被裁剪或终止，不能依赖越界后的观测继续训练。
- 小车没有刹车动作，不能假设动作空间中存在减速/制动维度。
- 小车没有夹爪，不能假设抓取、释放、夹持信号存在。
- 坞的朝向参考轴未显式给出；货箱朝向误差按接口说明用 atan2(obs[11], obs[10]) 作为代理，但该代理依赖“对齐轴为 0”的假设。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation
control_type: continuous
morphology:
  body_type: wheeled cart with a freely moving fragile square crate
  actuator_type: longitudinal force along heading plus steering torque; no brake; no gripper
  contact_structure: non-prehensile pushing contact between cart and crate; rigid-body contacts with floor, partition and warehouse bounds
primary_objectives:
  - move the crate from near side to the delivery dock on the far side of the partition
  - make the crate fully inside the dock rectangle
  - align crate heading with the dock
  - make the crate nearly stationary inside the dock
  - hold the settled condition continuously for a short settling period
secondary_objectives:
  - avoid repeated hard impacts that damage the fragile crate
  - keep cart and crate inside the warehouse floor
  - navigate through the narrow partition opening without getting stuck
  - avoid excessive time consumption that causes truncation
main_failure_risks:
  - hard impacts during pushing cause crate damage and failure
  - no brake on cart leads to overshooting the dock or high settling speed
  - crate glides after release, making precise low-speed docking difficult
  - cart or crate leaves the warehouse rectangle
  - partition wall blocks progress or causes collisions
  - time budget exhausted before docking succeeds
  - proximity-only reward may cause hovering near dock without full delivery
```

## 10. 奖励职责拆解 reward_role_decomposition
主信号骨架判断：成功在观测空间中的投影是“货箱到坞偏移趋近 0、货箱朝向误差趋近 0、货箱速度趋近 0，并持续一小段稳定时间”。因此主信号算子族应以 delta(crate_to_dock_distance) 或 improvement 为核心，而不是仅用 proximity；proximity 单独使用会鼓励货箱停在坞附近但不完全进入、不对齐、不静止的悬停状态。辅助信号应围绕易碎接触门控、坞内 settling 稀疏事件、边界与静态障碍 hinge penalty 展开。小车速度本身不是主目标，因为小车可以移动而不移动货箱。

### 10.1 主职责 mandatory_roles
- role_id: crate_delivery_progress
  purpose: 奖励货箱到坞距离的持续减少，推动货箱跨越隔墙开口并接近坞。
  why_required: 主目标是货箱到达坞，必须有信号引导货箱空间位置向坞收敛。
  usable_signals: [obs[12], obs[13], next_obs[12], next_obs[13], 由二者计算的 crate_to_dock_distance]
  risks: 若只用 proximity 而非 delta，容易悬停在坞附近；若货箱卡住，delta 为零，需要其他探索或解卡信号辅助。

- role_id: gentle_contact_impact_control
  purpose: 抑制小车-货箱之间的高烈度接触，避免 3 次硬冲击导致失败。
  why_required: 货箱易碎，硬冲击是明确失败条件。
  usable_signals: [obs[14], obs[4], obs[8], obs[9], obs[6], obs[7], 由相对运动构造的冲击代理]
  risks: 真实 contact_impulse 与 hard_collision_count 不可读，只能做代理；过度惩罚正常推动会导致无法移动货箱。

- role_id: dock_settling_and_alignment
  purpose: 在货箱接近或进入坞后，鼓励完全进入、朝向对齐、低速静止，并维持稳定状态。
  why_required: 成功要求 |obs[12]| <= 0.024、|obs[13]| <= 0.030、朝向误差 < 30 度、货箱速度 < 0.05 m/s，并持续 10 步。
  usable_signals: [obs[12], obs[13], obs[10], obs[11], obs[8], obs[9], 跨 step 维护的派生稳定条件]
  risks: 若用稠密 proximity 代替完整成功条件，会奖励“靠近但未完全进入”；若要求持续推动，会与低速度 settling 冲突。

- role_id: out_of_bounds_avoidance
  purpose: 防止小车或货箱离开仓库地面矩形。
  why_required: cart_out_of_bounds 和 crate_out_of_bounds 都是失败终止。
  usable_signals: [obs[0], obs[1], 由 obs[0,1,2,3,6,7] 恢复的货箱世界坐标]
  risks: 应使用靠近边界才触发的 hinge penalty，不应把小车一直往中心拉，否则会干扰穿越隔墙开口和推箱。

### 10.2 条件职责 conditional_roles
- role_id: static_obstacle_avoidance
  condition_to_use: 当 sensor_front/left/right 显示隔墙或静态障碍接近，且小车需要穿过狭窄开口时使用。
  usable_signals: [obs[15], obs[16], obs[17]]
  risks: 传感器是接近度而非精确距离，也不是碰撞冲量；惩罚过强会阻碍通过开口。

- role_id: time_efficiency
  condition_to_use: 当训练显示大量 episode 因 time_limit 截断，且该职责不会鼓励高速冲撞时使用。
  usable_signals: [obs[18], next_obs[18]]
  risks: 催促快速完成会增加硬冲击和过冲风险；时间限制终止是 truncation，不是 success。

- role_id: action_smoothness
  condition_to_use: 当观测到转向剧烈振荡、货箱被反复甩动或接触不稳定时使用。
  usable_signals: [action[0], action[1], obs[5], next_obs[5]]
  risks: 任务没有明确要求最小动作或最小能耗；过强平滑会妨碍必要的推箱动作。

- role_id: terminal_success_bonus
  condition_to_use: 仅在能用 obs 构造出派生成功链时使用：货箱完全进入坞、朝向误差 < 30 度、货箱速度 < 0.05 m/s，并且跨 step 保持足够步数。
  usable_signals: [obs[12], obs[13], obs[10], obs[11], obs[8], obs[9], 内部维护的持续稳定计数器]
  risks: 派生条件不完整会产生假成功；成功终止立即结束，不能依赖成功后的时间累积奖励。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: cart_forward_velocity_as_main
  reason: 任务核心是货箱交付，不是小车奔跑；小车向前冲不等于货箱向坞移动。
  forbidden_or_missing_signals: []

- role_id: action_energy_penalty
  reason: 任务没有明确要求省能量/省燃料；action_energy 是 forbidden info 字段；惩罚动作幅度可能抑制必要的推箱力。
  forbidden_or_missing_signals: [action_energy]

- role_id: contact_impulse_direct_penalty
  reason: contact_impulse 和 hard_collision_count 均不可读，无法直接惩罚真实硬冲击。
  forbidden_or_missing_signals: [contact_impulse, hard_collision_count]

- role_id: proximity_only_dock_reward
  reason: 成功要求完全进入、朝向对齐、低速并持续稳定；仅奖励靠近坞会鼓励悬停收割。
  forbidden_or_missing_signals: []

- role_id: info_based_termination_penalty
  reason: termination_reason、is_success、cargo_inside_dock、stable_steps 等 info 字段被禁止用于生成奖励。
  forbidden_or_missing_signals: [termination_reason, is_success, cargo_inside_dock, stable_steps]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_delivery_progress | obs[12], obs[13], next_obs[12], next_obs[13] | 无精确进度计数器 | delta_signal, dense_state_signal, bounded_signal | 主信号应衡量货箱到坞距离减少，而不是单纯 proximity。 |
| gentle_contact_impact_control | obs[14], obs[4], obs[8], obs[9], obs[6], obs[7] | contact_impulse, hard_collision_count, 精确峰值法向冲量 | hinge_penalty, gate, bounded_signal | 用接触 + 相对速度构造冲击代理；只在超过安全范围时惩罚。 |
| dock_settling_and_alignment | obs[12], obs[13], obs[10], obs[11], obs[8], obs[9] | cargo_angle_error, cargo_inside_dock, stable_steps | indicator, sparse_event_bonus, terminal_bonus, bounded_signal, gate | 用派生条件判断完全进入、朝向对齐、低速；持续步数需跨 step 维护。 |
| out_of_bounds_avoidance | obs[0], obs[1], 派生货箱世界坐标 | 精确越界标记 | hinge_penalty, bounded_signal | 只在接近边界时惩罚；不要全局拉向中心。 |
| static_obstacle_avoidance | obs[15], obs[16], obs[17] | 精确距离、碰撞冲量 | hinge_penalty, bounded_signal | 用于隔墙/开口导航；避免过强惩罚阻碍通过。 |
| time_efficiency | obs[18], next_obs[18] | 剩余步数精确值不可直接读取，但可由 time_fraction 推断 | dense_state_signal, bounded_signal | 仅在截断频繁时加入；不要鼓励高速冲撞。 |
| action_smoothness | action[0], action[1], obs[5], next_obs[5] | action_energy | quadratic_penalty, bounded_signal | 条件职责；没有明确任务要求，慎用。 |
| terminal_success_bonus | obs[12], obs[13], obs[10], obs[11], obs[8], obs[9], 内部稳定计数器 | is_success, termination_reason, stable_steps | sparse_event_bonus, terminal_bonus, indicator | 必须标注为 derived_possible；不能读取 info 成功标记。 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 硬冲击导致货箱损坏 | episode 在接触后过早终止；接触时 obs[4] 高、obs[8]/obs[9] 突变大；obs[14] 频繁为 1 且伴随高速相对运动 | 加入接触冲击代理 hinge penalty 或 gate；在接触时限制推进速度；鼓励低速接近货箱。 |
| 过冲坞区 | obs[12], obs[13] 穿过零点后重新变大；货箱速度在坞附近仍高；episode 未成功但货箱曾接近坞中心 | 奖励 delta 而不是 proximity；在坞附近惩罚高速度；鼓励提前释放，让地面阻尼完成低速停靠。 |
| 无法满足 settling 条件 | obs[12], obs[13] 已很小，但货箱速度 > 0.05 m/s 或朝向误差 > 30 度；持续无法成功终止 | 加入坞内低速/对齐 shaping；引导最后阶段轻推或释放；避免要求持续推挤。 |
| 悬停或停滞在坞附近 | 货箱到坞距离下降后不再减少；|obs[12]| > 0.024 或 |obs[13]| > 0.030；obs[18] 持续增大 | 使用 delta 主信号和稀疏成功 bonus；加入时间效率条件职责；避免 proximity-only 奖励。 |
| 隔墙开口卡住或撞墙 | obs[15]/obs[16]/obs[17] 持续高；小车位置变化小；货箱进度停滞 | 加入静态障碍接近 hinge penalty；鼓励通过开口的路径；不要过度惩罚靠近墙。 |
| 小车或货箱出界 | obs[0]/obs[1] 接近 ±1；派生货箱世界坐标接近仓库边界；episode 提前结束 | 加入边界 hinge penalty；在接近边界时抑制继续推进。 |
| 小车打转或转向振荡 | obs[5] 角速度幅值高且频繁变号；obs[4] 低；货箱到坞距离不下降 | 条件性加入转向平滑或角速度 hinge penalty；检查 steer 动作是否过大。 |
| 货箱被推离目标且无法追回 | obs[12]/obs[13] 增大；小车与货箱相对位置 obs[6]/obs[7] 不利；货箱速度方向远离坞 | 增加向坞 delta 信号；引导小车绕到货箱后方；避免仅奖励接触。 |
| 时间截断频繁 | obs[18] 接近 1.0；episode 以 truncation 结束；货箱未完全交付 | 条件性加入时间效率信号；提升主进度信号密度；检查是否陷入局部循环。 |
