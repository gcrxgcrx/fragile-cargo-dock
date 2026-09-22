# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D飞行器轨迹优化任务。一个带有双支撑腿的机体从视口顶部中央附近开始，具有随机初始力。主要目标是使机体尽快到达视口中央的目标着陆垫，并在其上稳定停靠，同时尽量少使用引擎推力。智能体应学会**接近目标 → 减速 → 保持竖直姿态 → 实现平稳、安全的接触并最终静止**。附属目标包括节省燃料和快速完成，但二者服务于主要目标，不与主要目标构成对等冲突的多目标结构。

## 2. 任务类型选择
**selected_route_id:** `navigation_goal_reaching`  
**confidence:** high  
**reason:** 任务的核心要求是“到达并停靠在中央目标垫上”，观察空间包含相对目标的连续位置，目标区域明确，主要的成功条件与位置、速度、姿态有关，符合导航与目标到达任务族。附属的省燃料、快速性属于典型的性能优化，不改变主任务性质。

## 3. 观察空间 observation_space
- **type:** Box（连续向量）
- **shape:** (8,)
- **dtype:** 根据环境实现推断为 float32
- **各维含义列表：**
  - obs[0]: **x_position** – 机体相对于目标垫中心的水平坐标（纵向），奖励可用：true
  - obs[1]: **y_position** – 机体相对于目标垫高度的垂直坐标（高度），奖励可用：true
  - obs[2]: **x_velocity** – 水平线速度，奖励可用：true
  - obs[3]: **y_velocity** – 垂直线速度，奖励可用：true
  - obs[4]: **body_angle** – 机体倾角（弧度），0 代表竖直向上，奖励可用：true
  - obs[5]: **angular_velocity** – 角速度，奖励可用：true
  - obs[6]: **left_support_contact** – 左支撑腿是否与目标垫接触（1.0=接触，0.0=未接触），奖励可用：true
  - obs[7]: **right_support_contact** – 右支撑腿是否与目标垫接触（1.0=接触，0.0=未接触），奖励可用：true

所有八个维度均可作为奖励信号的来源。

## 4. 动作空间 action_space
- **type:** Discrete
- **n:** 4
- **动作含义：**
  - action 0: **no_engine** – 不点火，任机体惯性运动和受重力/外力影响
  - action 1: **left_orientation_engine** – 点燃左侧姿态发动机，产生使机体逆时针旋转的力矩（同时可能产生微小推力）
  - action 2: **main_engine** – 点燃主发动机，产生向上的推力用于减速或升力
  - action 3: **right_orientation_engine** – 点燃右侧姿态发动机，产生使机体顺时针旋转的力矩（同时可能产生微小推力）

动作空间为离散的四选一，无连续量。

## 5. step 与终止条件分析
### 5.1 终止模式
根据提供的 `termination_conditions`，有三种情况会导致 `terminated = True`：

1. **crash_or_body_contact** — 机体碰撞地面或底部结构，典型失败。
2. **horizontal_position_outside_viewport** — 机体水平位置超出视口边界，典型失败。
3. **body_not_awake_or_settled** — 机体不再“活跃”（速度、角速度极低，且可能已达到稳定状态）。该条件代表物理稳定，可能是成功（若停在目标垫上且姿态好），也可能是失败（若偏离目标或倒在地上）。

由于成功和失败均可能触发 `body_not_awake_or_settled`，且官方 step 没有提供任何额外 success 标志（info 为空字典），因此**不存在显式的成功或失败信号**，仅靠该终止条件不能直接区分成功/失败。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false
- **explicit_failure_flag_available:** false
- **allowed_info_fields:** 无（`info` 返回空字典 `{}`，无额外字段）
- **forbidden_or_uncertain_info_fields:** info 中不存在任何可依赖字段，禁止假设存在 `success`、`failure`、`termination_reason` 等字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：当前步的观察向量（8维）
- `action`：选取的动作（0,1,2,3）
- `next_obs`：执行动作后的观察向量（8维）
- `info` 中明确允许的字段：无（因为该环境 info 恒为空）
- `training_progress`：仅当 prompt 明确允许时才可用，此处未提及，默认不建议依赖

**禁止使用：**
- `original_reward`（官方已掩码）
- 任何未在任务描述中声明的 info 字段
- 任何未在观察空间中声明的数据维度

## 7. 可用于奖励函数的信号
- **位置信号：** `next_obs[0]` (x)、`next_obs[1]` (y) — 接近目标时绝对值应趋于零。
- **速度信号：** `next_obs[2]` (x_vel)、`next_obs[3]` (y_vel) — 目标附近速度应接近零。
- **姿态信号：** `next_obs[4]` (body_angle) — 竖直姿态时角度为0（或接近0），可惩罚大偏离。
- **角速度信号：** `next_obs[5]` (angular_velocity) — 目标附近应接近零。
- **接触信号：** `next_obs[6]` (左接触)、`next_obs[7]` (右接触) — 可表征着陆状态，用于鼓励平稳接触。
- **动作/引擎信号：** `action` — 可用于鼓励少用引擎（No.0 无引擎），惩罚使用主引擎或姿态引擎以节省燃料。
- **其他：** 可通过 `obs` 与 `next_obs` 的差分计算位置变化、速度变化等。

## 8. 不确定或不可用的信号
- **环境提供的显式成功/失败标志：** 不存在。
- **目标垫位置：** 默认认为目标垫中心位于 (x=0, y=0)，但并未显式给出，仅由 obs[0] 和 obs[1] 是相对于目标的偏移推断。若实际目标位置有偏移，则不成立，但可安全假定相对坐标已归零。
- **机体质量、推力大小等物理参数：** 不在观察内，不可用。
- **风或其他外部干扰：** 源码中提及 wind 但屏蔽，不可用。
- **全局时间步数：** 不可用，除非 training_progress 被允许，否则不建议在奖励中引入基于时间/步数的奖励。
- **termination 原因判别：** 无法从 info 获得，禁止假设某终止条件对应成功。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact   # 接近目标并低速、稳定接触/停靠
control_type: discrete                             # 动作为离散四选一
morphology:
  body_type: 2D_lander_with_two_legs               # 双支撑腿着陆器
  actuator_type: main_engine + left/right_orientation_engines
  contact_structure: two_point_contact (left_leg, right_leg)
primary_objectives:
  - arrive_at_target_pad: minimize final distance to (x=0, y≈目标垫高度)，安全停靠
  - stable_settlement: final velocities and angular velocity near zero, upright orientation
secondary_objectives:
  - fuel_efficiency: minimize engine usage (non-zero actions)
  - time_efficiency: reach the pad quickly (隐含，但无直接步数信号可用)
main_failure_risks:
  - crashing_into_ground: body contact with ground or other obstacles
  - drifting_out_of_viewport: horizontal position beyond bounds
  - unstable_landing: high velocity contact or tilted body causing failure even after pad touch
  - overshoot_or_undershoot: poor approach dynamics leading to failure before settlement
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id:** `distance_to_target`
  - **purpose:** 引导飞行器向目标垫移动，每一步缩小相对距离。
  - **why_required:** 主要目标是到达目标位置，没有位置引导将无法收敛到垫附近。
  - **usable_signals:** `next_obs[0]` (x), `next_obs[1]` (y)
  - **risks:** 单纯惩罚距离可能忽略速度控制，导致以高速撞击垫或超出；需要配合速度与姿态惩罚。

- **role_id:** `soft_landing_velocity`
  - **purpose:** 在接近目标时降低线速度和角速度，实现平稳停靠。
  - **why_required:** 若仅引导位置，飞行器可能以高速撞击垫，触发 crash 或因不稳定导致失败，无法完成“稳定停靠”。
  - **usable_signals:** `next_obs[2]`, `next_obs[3]`, `next_obs[5]` (x_vel, y_vel, angular_vel)
  - **risks:** 过早施加强力速度惩罚可能阻止飞行器从高处下落，需在近距离或接触附近加强。

- **role_id:** `upright_orientation`
  - **purpose:** 保持机体竖直，防止因姿态过大导致碰撞或无法稳定。
  - **why_required:** 任务明确要求“保持稳定姿态”；倒置或大倾角接触垫通常导致失败。
  - **usable_signals:** `next_obs[4]` (body_angle)
  - **risks:** 角度稳定性可能需配合角速度惩罚一同使用。

### 10.2 条件职责 conditional_roles
- **role_id:** `fuel_penalty`
  - **purpose:** 鼓励减少引擎使用，节省燃料。
  - **condition_to_use:** 当飞行器已足够接近目标（例如 |x|<阈值、y 较低）或者接触垫后加强，早期可不施加或极轻微，避免干扰下降探索。
  - **usable_signals:** `action` (非0动作受惩罚)
  - **risks:** 若全程施加较强的引擎惩罚，可能导致飞行器不敢点火而直接坠落失败，必须与距离和速度职责平衡。

- **role_id:** `contact_reward`
  - **purpose:** 奖励双腿接触垫并保持稳定，强化安全着陆行为。
  - **condition_to_use:** 当 `next_obs[6]` 和 `next_obs[7]` 均为 1.0 时给予一次性或持续小奖励；也可结合速度低、角度小来提升奖励可靠性。
  - **usable_signals:** `next_obs[6]`, `next_obs[7]`, 速度、角度
  - **risks:** 如果单纯奖励接触而不检查速度和角度，可能鼓励猛烈撞击，应组合使用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id:** `time_step_penalty`
  - **reason:** 无法获取全局步数，training_progress 未被明确允许使用；即使可用，也不应在无训练进度参数时依赖。
  - **forbidden_or_missing_signals:** 全局时间步数。

- **role_id:** `success_oriented_bonus`
  - **reason:** 环境无显式成功标志，无法在奖励函数中区分成功终止。
  - **forbidden_or_missing_signals:** info 中无 success 字段。

- **role_id:** `exact_zoning_reward` (如同环数奖励)
  - **reason:** 目标垫周围无离散区域划分，观察无相关信息，不适用。
  - **forbidden_or_missing_signals:** 区域编号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `distance_to_target` | `next_obs[0]`, `next_obs[1]` | 无 | `dense_state_signal`, `bounded_signal` (e.g., -L2距离) | 可对水平和垂直分量分别加权，因垂直下降可能需单独引导。 |
| `soft_landing_velocity` | `next_obs[2]`, `next_obs[3]`, `next_obs[5]` | 无 | `quadratic_penalty` (速度惩罚), `bounded_signal` | 建议在接近垫或接触时增强，可与接触标志联动。 |
| `upright_orientation` | `next_obs