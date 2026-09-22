# 匿名环境理解卡片

## 1. 任务目标
智能体需要控制一个 2D 飞行器从视口上方初始位置出发，最终稳定停靠在中央目标平台上。  
**主目标**：安全、准确地降落在目标平台上并稳定停靠（settled）。  
**次目标**：在满足主目标的前提下尽量减少燃料消耗、缩短到达时间。  
**非任务目标**：单纯追求快、单纯节省燃料而不考虑着陆安全，都不可取。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达并稳定停靠在指定的目标平台，位置误差的收敛是最根本的成功条件；能耗与速度是附属优化项。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（通常连续空间默认）
- obs[0]: x_position，水平坐标（相对于目标平台中心），reward_usable: true
- obs[1]: y_position，垂直坐标（相对于平台高度），reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾斜角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作 0: no_engine，不点火
- 动作 1: left_orientation_engine，点燃一侧姿态发动机，产生偏航力矩
- 动作 2: main_engine，点燃主发动机，向下喷气提供向上推力（抵抗重力）
- 动作 3: right_orientation_engine，点燃另一侧姿态发动机，产生反向偏航力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - body_not_awake_or_settled：机体进入休眠或判定为已稳定停靠（可能结合接触和低运动状态），此条件终止可视为潜在成功。
- failure-like termination:
  - crash_or_body_contact：机体主体与地面碰撞或过于猛烈接触导致坠毁。
  - horizontal_position_outside_viewport：水平方向超出视口边界。
- ambiguous termination:
  - body_not_awake_or_settled 在没有足够上下文时也可能是失败（例如倒置卡死），但更常指向成功。
- truncation: 未提及，无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 任何假设的 "success"、"failure"、"landed"、"crash" 标志均不存在于 info 中

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段：无，info 固定为空，不可用
- training_progress 当前 prompt 未明确允许使用，默认不使用

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片（只能使用上述 8 维观察）

## 7. 可用于奖励函数的信号
- position: next_obs[0]（x 误差），next_obs[1]（y 误差）
- velocity: next_obs[2], next_obs[3]
- orientation: next_obs[4]（角度），next_obs[5]（角速度）
- contact: next_obs[6], next_obs[7]
- action/engine: action 取值 0/1/2/3（可推导是否开主发动机、姿态发动机）

## 8. 不确定或不可用的信号
- 明确的成功标志（无 info 可用）
- 目标平台的绝对高度或视角外边界（只能从相对位置推断）
- 燃料剩余量（无燃料观测量，只有动作是否可以推导，但可能环境是有燃料限制的，只是观测中没有给出；该信号不可用）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs
  actuator_type: one main downward thruster + two rotational thrusters
  contact_structure: left and right leg touch sensors
primary_objectives:
  - reduce horizontal and vertical distance to target pad to zero
  - achieve near-zero velocities and near-zero angle to safely settle
  - activate both leg contacts while stationary on target
secondary_objectives:
  - minimize fuel consumption (avoid unnecessary thruster actuation)
  - complete landing as fast as possible (but not at expense of safety)
main_failure_risks:
  - excessive impact velocity causing crash
  - large tilt leading to body/legs contact and crash
  - overshooting the pad horizontally and going out of bounds
  - excessive hovering and fuel wastage without progress toward target
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target
  purpose: 惩罚与目标平台的相对位置误差，引导飞行器靠近平台
  why_required: 到达目标是首要任务，位置误差必须趋于零
  usable_signals: [next_obs[0], next_obs[1]]（x、y 偏差）
  risks: 过度奖励仅可能使 agent 快速但鲁莽地接近，需要与其他职责平衡

- role_id: soft_landing_dynamics
  purpose: 在接近目标且接近触地时，约束线速度、角度和角速度，确保平稳着陆
  why_required: 仅接近不足以成功，必须减速至安全范围、姿态接近垂直，才能避免 crash
  usable_signals: [next_obs[2], next_obs[3], next_obs[4], next_obs[5]]
  risks: 过早施加强减速约束可能导致不敢下降，需要根据垂直高度逐步激活

- role_id: safe_contact_encouragement
  purpose: 当双腿均接触平台且姿态、速度满足成功标准时给予正向确认（塑造成功态）
  why_required: 成功最终条件是稳定停靠，无接触则无法判定着陆
  usable_signals: [next_obs[6], next_obs[7]] 结合位置/速度/角度
  risks: 可能诱导 agent 在非目标位置触发假接触，需与位置约束联合判断

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  condition_to_use: 当任务进行到中后期，agent 已有能力稳定接近目标时，可适度惩罚不必要的发动机点火
  usable_signals: [action==2 (main engine on), action in {1,3} (side engines on)]
  risks: 早期过度惩罚会抑制探索和学习基本飞行，使用自适应权重或延迟引入

- role_id: time_to_land_bonus
  condition_to_use: 可选，仅在成功着陆前给与微小的时间效率奖励，但必须是附加的、不破坏安全性的
  usable_signals: 隐含在 episode length 或可在环境外统计，但不可单从 obs/action 获得，一般不纳入纯观测奖励
  risks: 极易导致 agent 选择危险高速接近，慎用或不用

### 10.3 慎用/禁用职责 avoid_roles
- role_id: constant_stay_alive_bonus
  reason: 任务需要着陆，而非存活，固定存活奖励会鼓励在空中漂浮，与着陆目标矛盾
  forbidden_or_missing_signals: 无对应生存状态

- role_id: angular_velocity_only_penalty（无高度条件）
  reason: 仅惩罚角速度而不考虑离地高度，会在远距离时过度抑制机动
  forbidden_or_missing_signals: 缺高度门控信号

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | next_obs[0], next_obs[1] | none | dense_state_signal (Euclidean or L1 distance), bounded_signal | 可使用负欧式距离，或分段 |
| soft_landing_dynamics | next_obs[2],next_obs[3],next_obs[4],next_obs[5], next_obs[1] (可作为门控) | none | quadratic_penalty (for velocity and angle), bounded_signal, conditional_gating | 使用高度门控，当 next_obs[1] 较小（接近平台）时激活 |
| safe_contact_encouragement | next_obs[6], next_obs[7], next_obs[0],next_obs[1],next_obs[2],next_obs[3],next_obs[4] | none | binary_condition_multiplier, dense_signal | 仅在双腿接触、位置速度角度均在安全范围内时给予一次性大奖励 |
| fuel_efficiency | action (0-3) | fuel consumption explicit signal | sparse_penalty (action==2 or action in {1,3}), time_decaying_weight | 可通过惩罚开引擎动作实现 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停犹豫不下降 | y 位置不减小，频繁小推力摆动 | 加强高度下降的奖励引导，减少对主发动机的惩罚 |
| 高速坠毁 | 末段 y_velocity 负值大，crash | 强化 touchdown 前的速度惩罚（以高度为条件） |
| 水平漂移出界 | next_obs[0] 绝对值一直增大至截断 | 增加水平位置惩罚权重或引入越界强烈负奖励 |
| 勉强触地但侧翻 | 双腿未同时接触，角度大，最终 crash | 强调双腿对称接触和姿态稳定奖励 |
| 过度使用主发动机 | 长时间 action==2 | 调节燃料惩罚强度，并检查是否需要中期激活 |