# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个带推进器的 2D 刚体（初始在顶部附近并带有随机初始力）到达并稳定停靠在中央目标着陆区域（target pad）。稳定停靠要求接近水平位置、极低速、竖直姿态并保持双支撑接触。

次要目标：在保证安全停靠的前提下，尽可能快地完成停靠，同时尽可能减少引擎推力使用（节省燃料）。

不应混淆的目标：并非单纯的位置追踪或连续前进，着陆后的“存活/平衡”是结果而非持续目标，核心仍是到达并安全停靠。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: medium
reason: 任务核心是到达目标位置并稳定停靠（goal with settling），没有持续生存要求，不属于存活平衡；动作空间为离散推力控制，非连续步态驱动，不属于 locomotion_continuous_control；物体操作不存在，不属于 manipulation；没有强安全约束下的驾驶场景，不属于 autonomous_driving_safety；目标明确（着陆区域），不属于稀疏探索；虽然同时存在燃料与速度的次级目标，但它们服务于主目标且权重通常不冲突（慢而省油，快而费油，但安全着陆始终优先），因此不属于多目标冲突型任务。最接近的即为导航式目标到达。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position（相对目标 pad 的水平坐标），reward_usable: true
- obs[1]: y_position（相对 pad 高度的垂直坐标），reward_usable: true
- obs[2]: x_velocity（水平线速度），reward_usable: true
- obs[3]: y_velocity（垂直线速度），reward_usable: true
- obs[4]: body_angle（机体角度，0 表示竖直向上），reward_usable: true
- obs[5]: angular_velocity（角速度），reward_usable: true
- obs[6]: left_support_contact（左支撑触点状态，1.0=接触，0.0=未接触），reward_usable: true
- obs[7]: right_support_contact（右支撑触点状态，1.0=接触，0.0=未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（无推力）
- action 1: left_orientation_engine（激活左侧姿态控制引擎，产生旋转力矩，可能带微小线加速度）
- action 2: main_engine（激活主引擎，产生向上的推力，同时有不大的线加速度）
- action 3: right_orientation_engine（激活右侧姿态控制引擎，产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体静止并被认为已停靠，其同时要求接触地面且极低动能，隐含成功着陆）
- failure-like termination: crash_or_body_contact（剧烈碰撞或错误接触，如高速撞击地面或翻倒）、horizontal_position_outside_viewport（水平飞出边界）
- ambiguous termination: body_not_awake_or_settled 本身是成功信号，但 crash_or_body_contact 中可能包含“部分着陆但未稳定”的情形，但仍被归为失败终止
- truncation: 未提供明确的步数限制，但存在默认 horizon（例如 1000 步），在实际环境中可视为非失败截断，但本说明未提供该信息，按通常情况作为潜在截断处理（无 info 字段）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（无 info 字段，也没有显式 success 标志）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 在 step 中返回空字典 {}）
- forbidden_or_uncertain_info_fields: 一切 info 字段均不允许使用，因为源中未定义

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（前一步或当前步的观察，取决于函数调用时序，通常为当前步执行前的状态，或者某步之后的 next_obs；习惯上设计奖励时使用 next_obs 来衡量结果）
- action（当前步执行的动作，可用于惩罚推力使用）
- next_obs（执行动作后的新观察，用于评估着陆质量、接近程度等）
- training_progress 仅当 prompt 明确允许或需要课程式调节时才使用，否则保持为 0.0
- info 中不允许任何字段，因为其内容为空且未被声明

禁止使用：
- original_reward（官方奖励，严禁复制）
- official_reward
- info 中的任何未声明字段
- 任何未被上述声明的 obs 切片（但所有 8 维均合法可用）

## 7. 可用于奖励函数的信号
- position: x_position, y_position（相对目标 pad 的水平与垂直位移）
- velocity: x_velocity, y_velocity
- orientation: body_angle, angular_velocity
- contact: left_support_contact, right_support_contact（布尔化 1.0/0.0）
- action/engine: action 值 0/1/2/3 可用于判断是否使用推力
- other: 可基于上述信号合成距离、速度幅值、角度绝对值、是否着陆成功等衍生量

## 8. 不确定或不可用的信号
- 无显式的 success/failure 标签，必须通过观察推断成功着陆（如 low y_position, both legs contact, small speed, small angle）
- 无剩余时间或步数信息（除非 training_progress 隐含，但谨慎使用）
- 无与地形碰撞类型的直接标签，例如“leg contact”与“body crash”的区别无法直接获得，只能通过速度、角度和触点状态区分（如大的速度、无触点则为坠毁）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_two_leg_supports
  actuator_type: main_vertical_thruster_and_two_side_orientation_thrusters
  contact_structure: two_point_contact_on_bottom_legs
primary_objectives:
  - 精确停在目标 pad 上方（x,y 接近 0,0）
  - 低速垂直着陆（y_velocity ≈ 0，x_velocity ≈ 0）
  - 保持近似竖直姿态（body_angle ≈ 0，angular_velocity ≈ 0）
  - 两支撑点同时接触地面（left_contact and right_contact）
secondary_objectives:
  - 最小化引擎使用次数或总推力消耗
  - 尽可能快地完成着陆（隐含于时间惩罚或基于步数的奖励）
main_failure_risks:
  - 猛烈撞击地面导致 crash（高速，大角度）
  - 水平飘出可视区域
  - 仅单腿接触或翻倒（姿态失衡）
  - 在目标区域外低速停靠（偏离 pad，但未终止，需靠边界条件处理）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity
  purpose: 鼓励 agent 向目标 pad 中心移动，缩小 x,y 位置偏移
  why_required: 到达目标是核心任务，必须提供连续引导信号，否则稀疏奖励难以学习
  usable_signals: [x_position, y_position] （可使用距离的负值或关于位置的递减函数）
  risks: 若仅用距离，可能鼓励 agent 快速冲向中心但无减速，导致坠毁；需与速度惩罚配合

- role_id: safe_landing
  purpose: 保证着陆时刻的低速、竖直、双足着地
  why_required: 成功终止信号无法直接获取，必须通过密集奖励塑造安全着陆条件，且最终“settled”终止只能由这些条件达成
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact, y_position]
  risks: 若在未接近地面时就对速度/角度施加过大惩罚，会抑制 explorer；需配合高度条件或仅在靠近地面时激活（通过 y_position 较小或接触逐渐引入）

- role_id: stability_boost
  purpose: 使 agent 在接触地面后保持平衡，防止弹起或翻倒
  why_required: 部分成功着陆可能因速度残留再次离地，需要持续抑制角速度和姿态偏转
  usable_signals: [angular_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 若接触前就开始强加，可能导致不敢接近地面；可仅在至少一腿接触时施加

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  purpose: 减少不必要的引擎使用
  condition_to_use: 在训练中后期或当 agent 已能稳定着陆时引入，或者作为较小的常驻惩罚以促进节能
  usable_signals: [action] （action=1/2/3 时给予微小负奖励）
  risks: 若初始权重过高，会鼓励 agent 完全不使用引擎或自由落体，导致 crash

- role_id: fast_landing_bonus
  purpose: 鼓励在成功前提下尽快着陆
  condition_to_use: 仅在成功着陆那一步给予与剩余时间或步数相关的奖励（通过 training_progress 或自定义计数器），需谨慎避免让 agent 选择危险高速着陆
  usable_signals: [可结合成功判断和步数信息，但需在外部维护步数计数，或通过 distance to target + velocity 推断]
  risks: 若奖励形状设计不当，会牺牲安全性；若无可靠的剩余步数访问，可将“快速”视为在整个 trajectory 上用稀疏成功奖励间接促进，不作为单独密集项

### 10.3 慎用/禁用职责 avoid_roles
- role_id: continuous_survival
  reason: 本环境为一次性着陆任务，不是持续平衡任务；只要着陆后 terminated，无需存活奖励。若有，会误导 agent 为保持存活而悬浮不降落。
  forbidden_or_missing_signals: 无持久生存需求

- role_id: info_based_success
  reason: 环境未提供 info 中的 success 或 failure 标签，无法直接使用
  forbidden_or_missing_signals: [info["success"], info["failure"]] 均不可用

- role_id: contact_only_reward
  reason: 仅依赖接触标志可能导致 agent 在任何位置粗暴撞地；必须结合位置与速度条件
  forbidden_or_missing_signals: 仅接触标志不足以定义成功

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | — | dense_state_signal, bounded_signal (e.g., exp(-dist)) | 将距离映射到正奖励，越近越高 |
| safe_landing | y_position, x_velocity, y_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact | — | conditional_penalty (仅在接近地面或接触时生效), quadratic_penalty, velocity_penalty | 可用 y_position 阈值激活，双足同时接触且低速小姿态给正奖励 |
| stability_boost | angular_velocity, body_angle, left_support_contact, right_support_contact | — | bounded_signal (penalty on |body_angle|, |angular_velocity| when any leg contact) | 防止接触后翻倒 |
| fuel_efficiency_penalty | action | — | action_cost (fixed penalty per engine fire) | 对所有非零动作给予小负奖励 |
| fast_landing_bonus | 需要成功检测 + 时间或步数 | 无内置稀疏成功标记或时间信息 | sparse