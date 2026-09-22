# 匿名环境理解卡片

## 1. 任务目标
主任务：控制一个2D刚体（搭载推力器）从视口顶部中央附近出发，**到达并稳定、安全地停靠在画面中央的目标垫上**。  
次任务：在满足主任务的前提下，**尽可能少地使用引擎推力**（省燃料），并**尽快完成**。  
不应混淆的目标：纯粹的燃料最小化或最短时间不应牺牲安全接触与姿态稳定，着陆必须平稳。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 任务的核心是到达设定的目标垫并稳定停靠，附属目标为燃料与时间优化，符合导航目标到达族的典型特征。由于要求“软接触”“稳定姿态”“降低速度”，动力学子类型进一步细化。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 （推断，标准连续空间）
- obs[0]: `x_position`，水平相对坐标（可能是到目标垫中心的横向距离），可用于奖励（希望→0）
- obs[1]: `y_position`，垂直相对坐标（相对于垫的高度），可用于奖励（希望→0）
- obs[2]: `x_velocity`，水平线速度，可用于奖励（希望→0）
- obs[3]: `y_velocity`，竖直线速度，可用于奖励（着陆时希望→0或很小负值，但通常希望为0）
- obs[4]: `body_angle`，刚体朝向角度，可用于奖励（希望→0，保持直立）
- obs[5]: `angular_velocity`，角速度，可用于奖励（希望→0）
- obs[6]: `left_support_contact`，左侧支撑接触标志，0或1，可用于奖励（着陆成功时期望=1）
- obs[7]: `right_support_contact`，右侧支撑接触标志，0或1，可用于奖励（着陆成功时期望=1）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`（无推力，依靠惯性滑行）
- action 1: `left_orientation_engine`（向左定向推力，可能影响角速度或横向平移）
- action 2: `main_engine`（主推力，通常向上喷气，产生主要向上加速度，对抗重力）
- action 3: `right_orientation_engine`（向右定向推力，与左引擎对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled`  
  含义：刚体不再“醒着”（动能很低、角度稳定、接触垫子后进入休眠或标记为已停靠）。推测当成功着陆并稳定后触发。
- **failure-like termination**:
  - `crash_or_body_contact`：身体发生异常碰撞（可能并非目标垫，例如撞到地面、墙壁或其他部分）
  - `horizontal_position_outside_viewport`：水平位置超出允许范围，飞出视野
- **ambiguous termination**: 无
- **truncation**: 未给出 max steps 信息，可能不存在时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 为空，无 `success` 字段）
- explicit_failure_flag_available: **false** （无 `failure` 字段）
- allowed_info_fields: **无** （`info = {}`，禁止使用任何额外信息）
- forbidden_or_uncertain_info_fields: 任何未在 source 中声明为可用的字段，包括 `success`, `failure`, `terminal_observation` 等

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：动作前的观察数组（8维）
- `action`：执行的动作（整数值0-3）
- `next_obs`：动作后的观察数组（8维）
- `info`：仅允许空的字典，不可用任何字段
- `training_progress`：只有显式声明时可用（本例未允许，因此不可依赖）

**禁止使用：**
- `original_reward`（官方奖励已屏蔽，不可重构）
- 任何未声明的 `info` 字段（如 `success`, `failure`）
- 任何未在以上列表中出现的变量或全局状态

## 7. 可用于奖励函数的信号
- **position**: `obs[0]` (x), `obs[1]` (y) — 可计算到目标垫的距离，鼓励趋近于0
- **velocity**: `obs[2]` (vx), `obs[3]` (vy) — 可惩罚绝对值以减慢速度，特别是接近目标时
- **orientation**: `obs[4]` (angle) — 可惩罚偏离直立的角度
- **angular velocity**: `obs[5]` — 可惩罚快速旋转
- **contact**: `obs[6]` (left), `obs[7]` (right) — 用于设计着陆成功奖励或接触条件，期望双侧均为1且速度很小
- **action / engine usage**: `action` 本身 — 可惩罚使用主引擎或所有引擎的情形，促进燃料效率
- **other**: 可从位置、速度、接触组合出复合信号（如“已着陆标志”：两个接触且速度/角度在阈值内）

## 8. 不确定或不可用的信号
- 成功/失败明确标记：不存在
- 任何与任务进度（除了当前obs）相关的信息：不可用
- 目标垫的世界坐标或绝对位置：仅知相对坐标
- 燃料剩余量：未在观察中给出
- 与重力、推力大小等物理参数：未提供，不能用于奖励缩放，必须通过试探或归一化

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs/contact points
  actuator_type: discrete thrusters (one main engine, two orientation engines)
  contact_structure: left_support_contact, right_support_contact
primary_objectives:
  - reach the target pad (minimize |x|, |y| to zero)
  - land softly (velocity near zero at contact)
  - stabilize orientation (angle → 0, angular vel → 0)
secondary_objectives:
  - minimize fuel consumption (penalize engine usage, especially main engine)
  - minimize time to land (possibly through small shaping, but not at cost of safety)
main_failure_risks:
  - crash into ground/walls due to excessive speed or angle
  - drift out of horizontal bounds
  - inability to cut velocity, leading to hard landing or bouncing off
  - overusing fuel and running dry (if fuel is limited, but not explicitly provided)
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: **goal_proximity**
  purpose: 驱动 agent 向目标垫移动，最终使相对坐标接近0
  why_required: 这是任务核心，没有此职责 agent 无法找到目标
  usable_signals: obs[0], obs[1] （x, y 距离），可用 next_obs 做增量奖赏
  risks: 过分奖励距离减少可能导致 agent 高速冲撞目标而无法减速

- role_id: **velocity_damping**
  purpose: 强制在接近目标时降低线速度，实现软着陆
  why_required: 过硬着陆会导致失败（crash 或接触不良），且环境要求 safe contact
  usable_signals: obs[2], obs[3] （vx, vy），可结合与目标的距离动态加权
  risks: 如果全程惩罚速度，可能在出发阶段抑制合理加速

- role_id: **orientation_stability**
  purpose: 保持身体直立，避免大角度导致接触失败或推力方向紊乱
  why_required: 两个支撑点需要同时接触，角度太大只会单侧接触或翻倒
  usable_signals: obs[4] (angle), obs[5] (angular_velocity)
  risks: 过于严苛可能抑制必要的姿态调整（如利用方向引擎微调）

- role_id: **safe_contact**
  purpose: 确保两个支撑腿/接触点都着垫，且速度足够小，实现稳定停靠
  why_required: 这是最终成功的标志，环境很可能在 two contacts + settled 时自然终止并 success
  usable_signals: obs[6], obs[7] (contact flags), obs[2]/[3] (velocity)
  risks: 若接触要求过早引入，可能会鼓励 agent 在未对齐时就强行接触导致 crash

### 10.2 条件职责 conditional_roles
- role_id: **fuel_efficiency**
  condition_to_use: 整个 episode 始终可用，但应作为次要目标，不应牺牲主目标
  usable_signals: action (0为无消耗，1/2/3为消耗燃料), 主引擎可能燃料消耗更高或在某些场景被重点惩罚
  risks: 过度严厉的惩罚会使 agent 不愿启动引擎，导致无法到达目标

- role_id: **time_to_land** (optional)
  condition_to_use: 如果存在隐式任务时间限制或希望鼓励更早完成，可以使用小的每一步惩罚
  usable_signals: 每个 step 固定负值（但必须明确环境是否有 max steps 及是否希望 time shaping）
  risks: 时间压力大会导致 agent 牺牲软着陆和燃料效率

### 10.3 慎用/禁用职责 avoid_roles
- role_id: **sparse_success_bonus**
  reason: 没有明确的 success/failure flag，无法可靠触发；基于观测推断的“成功”可能有歧义，可能奖励到伪成功状态
  forbidden_or_missing_signals: info["success"], info["failure"]

- role_id: **progress_bonus_from_info**
  reason: info 为空，无外界进度信号
  forbidden_or_missing_signals: 任何 info 字段

- role_id: **exploration_bonus**
  reason: 任务非稀疏探索型，且无高维状态需要额外探索引导；盲目的探索可能造成 crash，得不偿失

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | obs[0], obs[1] | — | `dense_state_signal` (e.g. -sqrt(x²+y²)), `bounded_signal` (e.g. exp(-dist)) | 建议