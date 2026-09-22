# 匿名环境理解卡片

## 1. 任务目标
本任务是一个二维着陆问题：智能体控制一辆带主发动机和两个方向发动机的小车，从画面顶部中央附近开始（含随机初始冲量），尽快飞抵并平稳停靠在中央目标平台上。主要目标是到达目标并稳定着地；次要目标是尽量节省发动机推力（减少燃料消耗）并尽快完成。不应将姿态稳定或速度控制本身当作独立目标，它们是为安全着陆服务的。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心驱动是到达指定的目标位置（中央平台），着陆、减速、姿态稳定等均为完成到达/着陆的约束或附属优化。省燃料和快速到达是次要目标，不构成多目标冲突。因此归入“导航到达”类型，而非多目标任务。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32  
- 各维度含义（相对坐标系，原点为目标平台中心/高度）：
  - obs[0]: x_position – 相对于目标的水平坐标，reward_usable: true
  - obs[1]: y_position – 相对于平台高度的垂直坐标，reward_usable: true
  - obs[2]: x_velocity – 水平线速度，reward_usable: true
  - obs[3]: y_velocity – 垂直线速度，reward_usable: true
  - obs[4]: body_angle – 机体倾斜角，reward_usable: true
  - obs[5]: angular_velocity – 角速度，reward_usable: true
  - obs[6]: left_support_contact – 左支撑腿接触标志（0/1），reward_usable: true
  - obs[7]: right_support_contact – 右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0: no_engine – 不启动任何发动机
  - 1: left_orientation_engine – 启动左转向/姿态发动机
  - 2: main_engine – 启动主发动机
  - 3: right_orientation_engine – 启动右转向/姿态发动机

（动作是离散开关，无连续推力大小。每次 step 可选择执行一种发动机或待机）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` – 当身体进入休眠/稳定状态（可能表示已停稳在平台上）时终止，这最可能对应成功着陆。
- failure-like termination: `crash_or_body_contact` – 发生碰撞或身体其它部位不当接触（非支撑腿触地）时终止，代表坠毁；`horizontal_position_outside_viewport` – 飞出水平边界，失败。
- ambiguous termination: 无明确附加字段说明。
- truncation: 未提及 episode 截断，但在 RL 训练中超出最大步数会截断，此处不考虑。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无 success 键）
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，没有显式标志）
- forbidden_or_uncertain_info_fields: info 任何字段均不可用，不能去尝试猜测成功/失败标记。

注：终止原因本身在 compute_reward 函数外不可直接获知，只能通过 next_obs 的状态（如位置、速度、接触等）间接推断是否可能处于成功着陆状态。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 上一观察（8维）
- action: 所采取的动作（0~3 整数）
- next_obs: 下一观察（8维）
- info: 仅限其明确允许的字段（本环境为空，因此不可用）
- training_progress: 除非 prompt 明确说明允许，否则不可以使用

禁止使用：
- original_reward（被屏蔽，不得依赖）
- official_reward 或环境内部未暴露的任何变量
- info 的未知字段（本环境为空）
- 终止标志 done（函数参数列表不包含 done）
- 任何未在上述列表声明的 obs 切片或外部状态

## 7. 可用于奖励函数的信号
从 obs / next_obs / action 中可直接或间接使用的信号：
- 位置信号：
  - x, y（相对目标），可计算距离 `√(x²+y²)` 或分别处理
- 速度信号：
  - x_velocity, y_velocity，可计算合速度大小，尤其是垂直接近速度
- 姿态信号：
  - body_angle, angular_velocity，可用于维持竖直（angle≈0）
- 接触信号：
  - left_support_contact, right_support_contact（0/1），表示支撑腿是否着地，可作为着陆状态判断
- 动作信号：
  - 动作类型（0,1,2,3）可进行惩罚，因为每个非零动作代表一次发动机使用，消耗燃料。尤其可重点惩罚主发动机(2)，方向发动机(1,3)可较轻惩罚。
- 变化量信号：
  - 可由 obs→next_obs 计算速度变化、姿态变化来评估控制效果

## 8. 不确定或不可用的信号
- 与目标垫的接触力或碰撞强度（无直接观测）
- 剩余燃料量（无对应观测）
- 成功/失败标志（info 为空）
- 发动机推力大小（只有开关，无连续推力值）
- 距离边界的距离（只有位置本身，边界无显式给定，但可通过位置范围判断，但终止信息不包含边界值）
- 环境物理时间步长等参数（无法知道精确动态，奖励函数应基于观测相对变化）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete  # 离散动作空间（4个动作）
morphology:
  body_type: 2D 飞行器/着陆器，带两条支撑腿（着陆腿）
  actuator_type: 离散引擎（1个主推，2个姿态控制）
  contact_structure: 两条独立支撑腿接触检测
primary_objectives:
  - 将车身移动到目标平台且静止（x≈0, y≈0, 速度≈0, 角度≈0）
secondary_objectives:
  - 最少发动机启动次数（燃料节省）
  - 快速完成任务（步数少）
main_failure_risks:
  - 坠毁（身体非支撑腿触地）
  - 飞出水平边界
  - 着陆速度过大导致反弹或不稳定
  - 过度使用主发动机导致姿态失控
  - 角度过大翻倒
  - 燃料浪费（过度使用方向发动机）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: distance_to_target  
  purpose: 引导主体靠近目标平台  
  why_required: 核心任务，必须最小化位置误差  
  usable_signals: [x_position, y_position]  
  risks: 距离奖励若设计不当可能导致高速撞向目标，忽略减速；需与速度约束配合。

- role_id: soft_landing_velocity  
  purpose: 确保接触平台时速度接近零（特别是垂向速度），避免硬着陆  
  why_required: 任务要求“make safe contact”，且物理碰撞容易导致 crash  
  usable_signals: [x_velocity, y_velocity, left_support_contact, right_support_contact]  
  risks: 过早惩罚速度可能阻止智能体移动，必须根据是否接触或接近目标来条件化。

- role_id: upright_orientation  
  purpose: 保持机体竖直，防止倾斜过大导致翻倒  
  why_required: 稳定性需求，角度过大易引起 crash 且不利于着陆  
  usable_signals: [body_angle, angular_velocity]  
  risks: 在转向时可能为了方向控制需要短暂倾斜，过度惩罚会妨碍机动。

- role_id: fuel_penalty  
  purpose: 减少发动机使用，对应任务中的省燃料要求  
  why_required: 直接对应次目标，且防止无意义的大量推力  
  usable_signals: [action（离散值）]  
  risks: 若惩罚过重会阻碍探索，导致智能体不敢使用发动机；应配合距离/速度奖励进行调整。

### 10.2 条件职责 conditional_roles
- role_id: landing_contact_bonus  
  purpose: 当两条支撑腿均接触且速度足够小时给予完成奖励，强化成功着陆  
  condition_to_use: 仅在 next_obs 中左、右支撑腿均接触、速度很小、角度接近零、接近目标平台时激活  
  usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, x_position, y_position]  
  risks: 若条件过严或过松，可能导致稀疏奖励问题或错误奖励未稳定着陆的状态；需精细调参。

- role_id: crash_avoidance_penalty  
  purpose: 检测到即将发生坠落或 high-speed 接触时给予惩罚，强化安全性  
  condition_to_use: 在距平台很低但速度仍很高，或角度过大，或单边接触时施加额外惩罚  
  usable_signals: [y_position, y_velocity, angle, contacts]  
  risks: 可能产生误惩罚导致智能体过度谨慎、不敢着陆。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_penalty  
  reason: 步数惩罚与快速完成目标相关，但本环境未提供步数信息、也不应在奖励函数外累积计数；且可能干扰极简推力策略。如能基于 training_progress 等环境外信息引入，可考虑，但当前接口不允许。  
  forbidden_or_missing_signals: [步数或时间没有可靠观测，training_progress 未明确允许]

- role_id: success_flag_bonus  
  reason: 环境没有 success/failure 标志，无法安全使用。  
  forbidden_or_missing_signals: [无 success 标志]

- role_id: terminal_reward  
  reason: 不能利用 done 信息，无 done 参数，且 info 为空，无法获知 episode 结束。  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| distance_to_target | obs[0:2], next_obs[0:2] | — | distance: sqrt(x²+y²), 或分段线性/二次距离成本 | 需防止高速冲向目标，可与速度项联合 |
| soft_landing_velocity | next_obs[2:4], next_obs[6:7]（接触标志） | 无 | penalty on norm of velocity when contact is 1, or conditional penalty on y_velocity if near platform | 只应在双腿接触或接近平台时启用速度惩罚 |
| upright_orientation | next_obs[4], next_obs[5] | — | quadratic penalty: (angle)² + (angular_vel)² | 角度很小即可，无需精确0 |
| fuel_penalty | action | — | small constant per step if action != 0, heavier if action == 2 (main engine) | 可设为每个发动机动作固定代价，防止滥用 |
| landing_contact_bon