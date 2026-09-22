# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

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



# expert_reward_context.md

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





# 已知的奖励结构（本环境的作者奖励分项语义与权重；**优先于上文任何冲突规则**）

<!-- Generated by tools/extract_v9_structure_block.py — do not edit by hand.

     Provenance: the block below is extracted VERBATIM from
     `prompts/eureka_01_initial_reward_v9.md`, the prompt whose measured effect was
     4/60 -> 58/60 (runs/env_007/V9_STRUCTURE_FINDINGS.md). The CREATE pipeline injects it
     into the environment card so that CREATE and EUREKA receive the same reward
     information; see runs/env_007/CREATE_V9_SPEC_AND_STRUCTURE.md for the protocol and the
     declared asymmetry against the older CREATE baseline.

     Source sha256: bb3e15cd11343ee30f07aca73c35ca289d11ce9aa4da8b006ff272e9de8c16be
     Block sha256 : f3fca2aaeb9157bc8cd163fd605e87c921170f76ab073de928ea387eb99e0bac
-->

<!-- BEGIN v9 structure block (verbatim) -->

# 已知的奖励结构（本版本新增，**必须照此实现**）

本环境的作者奖励函数是已知的，其**分项语义与权重**如下。你的任务是：**用 `obs` 与 `action` 把它重写出来。**
（注意：这些是"设计结构"，不是"可以直接读取的字段"——你**不能**读 `info`，也拿不到任何未在观测中声明的量。）

| 分项 | 权重 | 语义 |
|---|---:|---|
| `approach_cargo` | **+1.0 / 米** | 本帧小车→货箱距离缩短了多少米（势函数形式） |
| `progress` | **+1.0 / 米** | 本帧货箱→泊位距离缩短了多少米（势函数形式） |
| `dock_enter` | **+5.0** | 货箱**首次**完全进入泊位容差时，一次性发放 |
| `roughness` | **−0.02 / (N·s)** | 接触冲量比例惩罚：撞得越狠罚越多（"轻拿轻放"） |
| `action_cost` | **−0.0005** | 动作平方和 |
| `time_cost` | **−0.002 / 步** | 每步固定时间成本 |
| `hard_hit` | **−0.5** | 单步内发生一次硬冲击的固定惩罚 |
| `terminal_success` | **+300** | 完成条件**连续 10 步**成立时发放一次 |
| `terminal_failure` | **−100** | 货箱或小车越界、或累计硬冲击 ≥3 次时发放一次 |

把这些分项重写成 `obs` + `action` 的表达式时，逐项的可行性与做法如下：

1. **`approach_cargo`**：小车→货箱距离（由 `obs[6]`, `obs[7]` 与姿态还原）**本帧的差**，按米计。
2. **`progress`**：货箱→泊位距离（由 `obs[12]`, `obs[13]` 还原）**本帧的差**，按米计。
   **两项都必须是有符号、对称的**：靠近给正分，远离给负分。
3. **`dock_enter`**：用模块级状态记录"是否曾经完全进入容差"，只发一次。
4. **`action_cost` / `time_cost`**：直接照写。
5. **`terminal_success`**：用模块级状态对"泊位内 + 对齐 + 慢"做**连续计数**，达到 10 次时发一次。
   注意本环境在满足时会**立即终止** episode，所以它天然最多只发一次。
6. **`terminal_failure`**：越界由 `obs[0]`/`obs[1]` 判断（`|x|` 或 `|y|` 超过约 1.05 即出界）。
7. **`roughness`（唯一需要代理的项）**：观测里**没有冲量**这一维。请构造一个可观测的
   "接近速度 × 接触"代理——例如
   `closing = obs[4]*3.0 - (obs[8]*3.0*obs[2] + obs[9]*3.0*obs[3])`，再取 `max(0, closing)`，
   仅在实际接触（`obs[14] > 0.5`）时乘以一个负系数。**注意两点**：
   该项是用来**抑制撞击**的，不是用来压制推进的——实测把它的量级做到与推进项相当，
   会让策略连动都不敢动（入坞率 0.00）。**它必须比推进项弱**，只在"撞得很快"时显著。

**优先级声明**：本节给出的分项表**优先于**下文任何与之冲突的通用规则；若下文的规则与本节冲突，以本节为准。

<!-- END v9 structure block -->



# Validation repair

具体错误：Reward v1 failed validation: 出现禁止模式: try: (record: runs\env_007\fragilecargo_create_v9\seed_0\iter_01\generation\validations\reward_v1.validation.json)

只修复代码合规问题，不重新分析环境，不改变原定奖励设计。直接输出完整合规的compute_reward函数。

# Invalid previous draft

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level episode state (no self, no extra function, no info) ----
    state = getattr(compute_reward, "_v1_state", None)
    if state is None:
        state = {
            "prev_cc": None,      # previous cart->crate distance (m)
            "prev_cd": None,      # previous crate->dock distance (m)
            "prev_t": None,       # previous time_fraction, for episode-reset detection
            "entered": False,     # has crate ever fully entered dock tolerance
            "stable": 0,          # consecutive settled steps inside dock
            "hard": 0,            # accumulated hard-hit proxy count
            "success_paid": False,
            "fail_paid": False,
        }
        compute_reward._v1_state = state

    # time_fraction is monotone inside an episode; a drop means a new episode started
    t_now = float(obs[18])
    if state["prev_t"] is not None and t_now < state["prev_t"] - 0.05:
        state["prev_cc"] = None
        state["prev_cd"] = None
        state["entered"] = False
        state["stable"] = 0
        state["hard"] = 0
        state["success_paid"] = False
        state["fail_paid"] = False
    state["prev_t"] = t_now

    heading_x = float(obs[2])
    heading_y = float(obs[3])

    # ---- geometry: cart -> crate (body-frame relative vector, metres) ----
    rel_x = float(obs[6]) * 3.0
    rel_y = float(obs[7]) * 3.0
    cc_now = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = float(next_obs[6]) * 3.0
    nrel_y = float(next_obs[7]) * 3.0
    cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    # ---- geometry: crate -> dock (metres) ----
    dock_x = float(next_obs[12]) * 5.0
    dock_y = float(next_obs[13]) * 4.0
    cd_next = (dock_x * dock_x + dock_y * dock_y) ** 0.5

    # ---- signed potential-difference signals ----
    if state["prev_cc"] is None:
        state["prev_cc"] = cc_next
    approach_cargo = 1.0 * (state["prev_cc"] - cc_next)
    state["prev_cc"] = cc_next

    if state["prev_cd"] is None:
        state["prev_cd"] = cd_next
    progress = 1.0 * (state["prev_cd"] - cd_next)
    state["prev_cd"] = cd_next

    # ---- one-shot reward on first full dock containment, from obs only ----
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    dock_enter = 0.0
    if inside and not state["entered"]:
        dock_enter = 5.0
        state["entered"] = True

    # ---- roughness proxy: contact x closing speed (obs has no impulse channel) ----
    cart_speed = float(obs[4]) * 3.0
    crate_along = float(obs[8]) * 3.0 * heading_x + float(obs[9]) * 3.0 * heading_y
    closing = cart_speed - crate_along
    contact = float(obs[14]) > 0.5

    roughness = 0.0
    hard_hit = 0.0
    if contact and closing > 0.0:
        roughness = -0.02 * closing            # deliberately much weaker than progress
        if closing > 1.5:                      # high-severity contact proxy
            hard_hit = -0.5
            state["hard"] += 1

    # ---- effort / time bookkeeping ----
    action_cost = -0.0005 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    time_cost = -0.002

    # ---- derived settling chain -> terminal_success (no info flag available) ----
    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > 0.866       # |heading error| < 30 deg, cos proxy
    if inside and aligned and crate_speed < 0.05:
        state["stable"] += 1
    else:
        state["stable"] = 0

    terminal_success = 0.0
    if state["stable"] >= 10 and not state["success_paid"]:
        terminal_success = 300.0
        state["success_paid"] = True

    # ---- derived failure chain -> terminal_failure ----
    crate_wx = float(obs[0]) * 5.0 + rel_x * heading_x - rel_y * heading_y
    crate_wy = float(obs[1]) * 4.0 + rel_x * heading_y + rel_y * heading_x
    out_of_bounds = (
        abs(float(obs[0])) > 1.05
        or abs(float(obs[1])) > 1.05
        or abs(crate_wx) > 5.25
        or abs(crate_wy) > 4.2
    )

    terminal_failure = 0.0
    if (out_of_bounds or state["hard"] >= 3) and not state["fail_paid"]:
        terminal_failure = -100.0
        state["fail_paid"] = True

    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "roughness": float(roughness),
        "action_cost": float(action_cost),
        "time_cost": float(time_cost),
        "hard_hit": float(hard_hit),
        "terminal_success": float(terminal_success),
        "terminal_failure": float(terminal_failure),
    }
    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + terminal_success
        + terminal_failure
    )
    return float(total_reward), components
```

# reward_v1 设计说明

**修复内容**：仅修复代码合规问题——上一版注释 `# ---- dock entry: ...` 中的字符串 `"entry:"` 会被禁止模式正则 `try:` 命中（`en**try:**`），触发 validation 的 "禁止模式: try:" 报错。现将该注释改写为 `# ---- one-shot reward on first full dock containment, from obs only ----`，不含 `try:` 子串。奖励设计、组件、权重、信号选择、模块级状态逻辑与上一版完全一致，未做任何改动。

**selected task_family / dynamics_subtype**
`manipulation_grasping` / `staged_manipulation`：连续控制的非 prehensile 推箱任务，成功判据落在货箱位姿 + 朝向 + 低速 + 持续稳定。

**selected reward roles**

| role_id | 类别 | v1 中的实现 |
|---|---|---|
| `crate_delivery_progress` | mandatory | `progress`（货箱→坞距离本帧减少）+ `dock_enter`（首次完全入坞一次性奖励） |
| `gentle_contact_impact_control` | mandatory | `roughness`（接触 × 接近速度代理）+ `hard_hit`（高烈度接触事件惩罚） |
| `dock_settling_and_alignment` | mandatory | `terminal_success` 的派生稳定链（入坞 + 对齐 + 低速，连续 10 步） |
| `out_of_bounds_avoidance` | mandatory | `terminal_failure` 的越界分支 |
| `approach_cargo` | 结构块要求 | 小车→货箱距离本帧减少，作为主信号引导项 |
| `time_efficiency` | conditional | 恒定 `time_cost = -0.002/步`（结构块给定） |
| `action_smoothness` | conditional | 极小 `action_cost = -0.0005 * Σa²`，不足以压制推箱 |

**role_to_signal_mapping（实际用到的信号）**
- `progress` / `dock_enter` / settling：`obs[12], obs[13], next_obs[12], next_obs[13], next_obs[8], next_obs[9], next_obs[10]`
- `approach_cargo`：`obs[6], obs[7], next_obs[6], next_obs[7]`
- `roughness` / `hard_hit`：`obs[14], obs[4], obs[8], obs[9], obs[2], obs[3]`
- `out_of_bounds`：`obs[0], obs[1]` + 由 `obs[2], obs[3], obs[6], obs[7]` 还原的货箱世界坐标
- 未使用：`obs[5], obs[15], obs[16], obs[17]`（留待后续迭代）

**formula operator**
- `approach_cargo` / `progress`：`improvement_delta`（有符号、对称，靠近给正、远离给负），不用 proximity-only。
- `dock_enter`：`sparse_event_bonus` + 一次性标志。
- `roughness`：`bounded_signal` 线性截断 `-0.02 * max(0, closing) * contact`，量级远小于 `progress`。
- `hard_hit`：`indicator`（closing > 1.5 且接触）→ −0.5。
- `terminal_success`：`joint_condition_proxy` 连续计数版（`inside × aligned × slow` 累加满 10 步一次）。
- `terminal_failure`：`indicator`（越界 ∨ 累计硬冲击 ≥ 3）+ 一次性标志。
- `action_cost` / `time_cost`：`quadratic_penalty` 与恒定步成本，权重照结构块。

**excluded roles 及原因**
- `cart_forward_velocity_as_main`：小车奔跑与货箱交付无因果关系。
- `action_energy_penalty`：`action_energy` 为 forbidden info。
- `contact_impulse_direct_penalty`：`contact_impulse`、`hard_collision_count` 不可读。
- `proximity_only_dock_reward`：会诱发坞旁悬停刷分。
- `info_based_termination_penalty`：相关 info 字段全部禁用。
- `static_obstacle_avoidance` 强形式：v1 不加静态障碍 hinge，避免阻碍通行。

**为什么仍使用 terminal_success / terminal_failure**
卡片 `explicit_success_flag_available = false`、`explicit_failure_flag_available = false`，因此二者完全从 `obs`/`next_obs` 派生（不入坞/越界判据见代码），并以一次性标志保证至多触发一次，且检测到 `obs[18]` 下降时整体重置。这是「已知奖励结构」块的强制要求，与卡片通用建议冲突时以结构块为准。

**留到后续迭代的职责**：静态障碍 hinge、显式时间效率加权、转向振荡抑制、更精确碰撞代理、边界前兆软惩罚。

**训练后应观察的 failure modes**
1. `roughness/hard_hit` 高频 → 高速撞击推箱，`hard` 常达 3；干预：改为 gate 乘到 `progress`。
2. `progress` 步均值≈0 且 `obs[18]` 走高 → 卡在隔墙开口/原地打转；干预：静态障碍 hinge。
3. 距离降到 ~0.2 m 后停滞 → delta 饱和，需坞附近低速/对齐 shaping。
4. `terminal_success` 从未触发但 `inside` 频繁真 → settling 门槛过难。
5. `terminal_failure` 主要来自货箱越界 → 核对体坐标系旋转符号。
6. 小车出界处反复触发 → 需边界前兆软信号。
```
